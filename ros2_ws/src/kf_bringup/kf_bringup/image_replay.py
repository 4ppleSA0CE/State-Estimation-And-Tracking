"""Live view: the real KITTI left colour camera, paced by the ego stream.

Started only by `full_pipeline.launch.py foxglove:=true`. Nothing in the gated path subscribes
to it, so this node can never change a verdict; with foxglove:=false it is not launched at all
and costs nothing.

    /kitti/image_02/compressed   sensor_msgs/CompressedImage, format="png", frame_id="camera"

The point of the node is the same as lidar_replay's -- context -- but for the human rather than
for the geometry: a viewer watching boxes drift in the 3D panel has no way to tell a tracker
failure from a car that genuinely left the scene. The road video answers that in one glance.

--------------------------------------------------------------------------------------------
No decoding, on purpose

KITTI ships image_02 as PNG on disk and CompressedImage carries exactly that: the file's own
bytes plus a format string. So the "conversion" here is a file read. No cv_bridge, no OpenCV,
no PIL -- decoding 1392x512 to raw and re-encoding would cost ~10 ms inside a 100 Hz callback
and produce a byte-for-byte worse version of what is already on disk. Foxglove renders
CompressedImage/png natively.

--------------------------------------------------------------------------------------------
Pacing (why a subscription and not a timer)

Identical to lidar_replay, and for the same reason: `rate_scale` stretches or compresses the
replay clock arbitrarily, so ANY wall-clock timer here would show the video from one moment
beside the ego pose from another -- and video is the one panel where a viewer would notice.
Frames are therefore driven by /ego/state, the replay's own 100 Hz heartbeat: on each ego
message we compute the nearest frame and publish it if it is not the one already sent. At most
one publish per frame, in order, at whatever rate the replay is actually running.

"Nearest" rather than "most recent": both are wrong by at most half a frame period instead of a
whole one. The lookup is a searchsorted over the MIDPOINTS between frame times, which is
exactly nearest-neighbour.

--------------------------------------------------------------------------------------------
Time base

The replay's t = 0 is the FIRST OXTS timestamp of the drive `cache_path` was built from, and
every stamp downstream is seconds from there. image_02/timestamps.txt is absolute UTC, so the
offset is real and the anchor drive matters: drive_0001 and drive_0009 were recorded ~6 minutes
apart, and anchoring to the wrong one puts every frame outside the replay window, which shows
up as playback frozen on frame 0 rather than as an error. Hence `oxts_timestamps` is a
parameter and not a guess.

Both files are differenced in seconds-of-day, never in absolute epoch seconds: epoch seconds
are ~1.3e9, where a float64 resolves ~2e-7 s, and the ns precision the rest of the pipeline
keys on would be quietly rounded away. All timestamps must be on the same date, which is
asserted rather than assumed.

drive_0001's camera leads its OXTS by 0.148 s, so its first two frames sit at NEGATIVE replay
time and are skipped -- see `_on_ego`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from kf_bringup.kitti_replay import _stamp_from_secs
from kf_msgs.msg import EgoState

IMAGE_FORMAT = "png"     # what KITTI ships; the bytes are passed through untouched

PARAMS: dict = {
    "image_dir": ("/workspace/data/kitti_raw/2011_09_26/"
                  "2011_09_26_drive_0001_extract/image_02"),
    # Defines t = 0 for the whole replay; must be the SAME drive pipeline_replay's cache was
    # built from, or the video is offset from the ego pose by a constant nobody can see.
    "oxts_timestamps": ("/workspace/data/kitti_raw/2011_09_26/"
                        "2011_09_26_drive_0001_extract/oxts/timestamps.txt"),
    "frame_id": "camera",
}


def _split_timestamp(line: str) -> tuple[str, float]:
    """'2011-09-26 13:02:25.594360375' -> ('2011-09-26', 46945.594360375).

    Seconds-of-day, not epoch seconds -- see the module docstring on float64 resolution.
    """
    date, _, clock = line.strip().partition(" ")
    hour, minute, second = clock.split(":")
    return date, int(hour) * 3600.0 + int(minute) * 60.0 + float(second)


def _read_timestamps(path: Path) -> tuple[list[str], np.ndarray]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    dates, secs = zip(*(_split_timestamp(line) for line in lines))
    return list(dates), np.asarray(secs, dtype=float)


class ImageReplay(Node):
    """Ego-paced KITTI image_02 publisher. All state is the index of the last frame sent."""

    def __init__(self) -> None:
        super().__init__("image_replay")

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = self.get_parameter

        image_dir = Path(str(p("image_dir").value))
        self._frame = str(p("frame_id").value)

        frame_dir = image_dir / "data"
        self._images = sorted(f for f in frame_dir.iterdir() if f.suffix == ".png")
        if not self._images:
            raise ValueError(f"{frame_dir} holds no .png frames")

        dates, secs = _read_timestamps(image_dir / "timestamps.txt")
        if len(dates) != len(self._images):
            raise ValueError(f"{image_dir}: {len(self._images)} image files but {len(dates)} "
                             f"timestamps; the two must index each other 1:1")
        oxts_path = Path(str(p("oxts_timestamps").value))
        date0, t0 = _split_timestamp(oxts_path.read_text().splitlines()[0])
        # Differencing seconds-of-day across a midnight boundary would be off by 86400 s and look
        # like "the camera never publishes". The KITTI drives are mid-afternoon, so this only
        # ever fires if the node is pointed at unrelated data.
        odd = {d for d in dates if d != date0}
        if odd:
            raise ValueError(f"image timestamps span date(s) {sorted(odd)} but the OXTS time "
                             f"base starts on {date0}; seconds-of-day differencing is invalid")
        self._t = secs - t0                       # replay seconds, the /ego/state time base
        # Nearest-neighbour lookup: searchsorted over the midpoints between consecutive frames.
        self._edges = 0.5 * (self._t[1:] + self._t[:-1])
        self._last = -1

        # RELIABLE, depth 2 -- the same trade lidar_replay makes, for the same reason and with the
        # same measurement behind it. A ~1 MB PNG is fragmented into ~700 UDP datagrams and the
        # container's SO_RCVBUF ceiling is net.core.rmem_max = 212992 B, so under BEST_EFFORT one
        # lost fragment discards the whole frame with no repair: 25/100 delivered in an isolated
        # 1 MB @ 10 Hz test, against 100/100 for RELIABLE. That is why the Foxglove image panel
        # looked dead while every small RELIABLE topic rendered fine.
        #
        # Depth 2 is load-bearing. KEEP_LAST(2) bounds the writer history, so a reader that falls
        # behind costs at most `max_blocking_time` (100 ms) and then the oldest frame is DROPPED --
        # still "drop a frame of a 10 Hz view", still bounded, never the unbounded stall inside the
        # /ego/state callback that BEST_EFFORT was originally chosen to avoid.
        image_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=2,
                               reliability=QoSReliabilityPolicy.RELIABLE)
        # RELIABLE to match eskf_node's /ego/state publisher -- best-effort here would not match
        # it and the node would receive nothing at all.
        ego_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                             reliability=QoSReliabilityPolicy.RELIABLE)

        self._pub = self.create_publisher(CompressedImage, "/kitti/image_02/compressed",
                                          image_qos)
        self.create_subscription(EgoState, "/ego/state", self._on_ego, ego_qos)

        # The one startup line. lidar_replay's equivalent is what made its clock bug diagnosable:
        # a span printed on the replay base immediately shows an anchor pointed at the wrong
        # drive, because every frame lands hundreds of seconds away from [0, run length].
        skipped = int(np.count_nonzero(self._t < 0.0))
        self.get_logger().info(
            f"image_replay up: {len(self._images)} images from {frame_dir} spanning "
            f"t = {self._t[0]:.3f} .. {self._t[-1]:.3f} s on the replay base "
            f"(anchor {oxts_path}, t0 = {date0} {t0:.9f} s-of-day); "
            f"{skipped} frame(s) predate t=0 and will be skipped; "
            f"format={IMAGE_FORMAT}, frame={self._frame}")

    # ------------------------------------------------------------------
    # Pacing.
    # ------------------------------------------------------------------
    def _on_ego(self, msg: EgoState) -> None:
        """Advance to the frame nearest this ego stamp, at most one publish per frame."""
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        index = int(np.searchsorted(self._edges, t))
        if index <= self._last:
            return                      # already sent, or the stream went backwards
        self._last = index
        # A frame recorded BEFORE the replay's t=0 has a negative time on this base, and a ROS
        # stamp is unsigned -- assigning one raises "The 'nanosec' field must be an unsigned
        # integer". That happens whenever the OXTS cache starts partway into the drive the images
        # came from (drive_0009_tail is the clean tail of drive_0009, so its first 19.4497 s of
        # frames predate t=0), and also for drive_0001's own first two frames, whose camera
        # leads its OXTS by 0.148 s. Those frames are genuinely outside the replay and must be
        # skipped, not clamped to zero: clamping would pile stale video onto the first instant.
        if self._t[index] < 0.0:
            return
        self._publish(index)

    def _publish(self, index: int) -> None:
        msg = CompressedImage()
        # The FRAME's own stamp, on the replay time base -- not the wall clock. Foxglove syncs
        # the image panel to the 3D scene by stamp, so a re-stamped frame would be shown beside
        # the wrong ego pose, which is the exact confusion the panel exists to remove.
        msg.header.stamp = _stamp_from_secs(float(self._t[index]))
        msg.header.frame_id = self._frame
        msg.format = IMAGE_FORMAT
        # The raw PNG file, byte for byte. See the module docstring on why nothing decodes it.
        msg.data = self._images[index].read_bytes()
        self._pub.publish(msg)
        self.get_logger().info(
            f"image {index}/{len(self._images) - 1} at t={self._t[index]:.3f} s, "
            f"{len(msg.data)} bytes", throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImageReplay()
    try:
        rclpy.spin(node)
    # The launch SIGINTs this node at teardown; rclpy signals that as ExternalShutdownException.
    # Unhandled it exits 1 with a traceback, and the launch logs a clean shutdown as a crash.
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
