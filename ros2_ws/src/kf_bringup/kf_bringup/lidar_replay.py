"""Stage 6 live view: the real KITTI Velodyne scans, paced by the ego stream.

Started only by `full_pipeline.launch.py foxglove:=true`. Nothing in the gated path subscribes
to /velodyne_points, so this node can never change a verdict; with foxglove:=false it is not
launched at all and costs nothing.

    /velodyne_points   sensor_msgs/PointCloud2, frame_id="velodyne", ~10 Hz
    /tf_static         base_link -> velodyne, from calib_imu_to_velo.txt

The point of the node is context: without it the 3D panel is boxes floating over an empty grid,
and there is no way to see that the ESKF's pose is putting the road where the road is.

--------------------------------------------------------------------------------------------
Pacing (why a subscription and not a timer)

A free-running timer drifts against the replay within seconds: `rate_scale` stretches or
compresses the replay clock arbitrarily (0.2 for a slow demo, 1.0 for wall clock), so ANY
wall-clock timer here would show the cloud from one moment beside the ego pose from another.
The scans are therefore driven by /ego/state, which is the replay's own 100 Hz heartbeat: on
each ego message we compute the nearest scan and publish it if it is not the one already sent.
That is at most one publish per scan, in order, at whatever rate the replay is actually running.

"Nearest" rather than "most recent": both are wrong by at most half a scan period instead of a
whole one, and at drive_0001's 13.3 m/s a scan period is 1.3 m of ego motion. The lookup is a
searchsorted over the MIDPOINTS between scan times, which is exactly nearest-neighbour.

--------------------------------------------------------------------------------------------
The base_link -> velodyne transform (the part that is plausible-looking when wrong)

`calib_imu_to_velo.txt` gives `R` (3x3, row-major) and `T` (3x1) for the IMU->Velodyne
direction, i.e. it maps COORDINATES one way:

    p_velo = R p_imu + T

`base_link` IS the KITTI IMU/body frame in this stack (x forward, y left, z up -- pipeline_replay
publishes OXTS body accelerations on /imu/data with frame_id="base_link"), so that file is
T_velo<-imu. A TransformStamped(frame_id=base_link, child_frame_id=velodyne) carries the
opposite direction -- child coordinates INTO parent coordinates, p_imu = R' p_velo + t' -- so it
is the INVERSE:

    R' = R^T
    t' = -R^T T = (+0.8105, -0.3071, +0.8027) m

Read that translation and it is self-checking against KITTI's published sensor-setup drawing:
the Velodyne sits 0.81 m FORWARD of, 0.31 m RIGHT of, and 0.80 m ABOVE the IMU box. The
Velodyne's documented height above the road is 1.73 m and the IMU's is 0.93 m; 1.73 - 0.93 =
0.80, the z entry above, derived independently.

Empirical check on scan 0 of drive_0001 (the one that catches a flipped sign): the ground
returns in the 4-15 m annulus land at z ~ -0.9 m in base_link, i.e. the road is just under a
metre below the IMU, as it must be. Using the FORWARD transform instead puts the same returns at
z ~ -2.5 m -- a road surface two and a half metres beneath the car, which renders as a perfectly
tidy, perfectly wrong scene.

--------------------------------------------------------------------------------------------
Time base

The replay's t = 0 is the FIRST OXTS timestamp of the extract drive (13:02:25.594360375), and
every stamp downstream is seconds from there. The Velodyne timestamps file is absolute UTC, so
the offset is real and worth getting right: the first sync scan is 0.357 s into the drive, which
is 4.7 m of ego motion -- visibly wrong if it were assumed to be zero.

Both files are read here and differenced in seconds-of-day, never in absolute epoch seconds:
epoch seconds are ~1.3e9, where a float64 resolves ~2e-7 s, and the ns-precision the rest of the
pipeline keys on would be quietly rounded away. All timestamps must be on the same date, which
is asserted rather than assumed.

--------------------------------------------------------------------------------------------
Scan source

drive_0001 ships two velodyne trees:

    ..._sync/velodyne_points/data/*.bin      108 scans, float32 [x, y, z, reflectance] * N
    ..._extract/velodyne_points/data/*.txt   111 scans, the same rows as ASCII

The default is the `_sync` tree: it is the motion-compensated, rectified product (the right thing
to draw beside a pose), and a 3 MB binary read is ~2 ms where re-parsing 120k ASCII lines is two
orders of magnitude slower -- inside a 100 Hz callback that difference is the difference between
a smooth view and a stuttering one. `.txt` is still accepted so the node also runs against the
`_extract` tree; the loader picks by suffix.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster

from kf_bringup.kitti_replay import _stamp_from_secs
from kf_msgs.msg import EgoState

BASE_FRAME = "base_link"
POINT_STEP = 16          # 4 float32: x, y, z, intensity
SECONDS_PER_DAY = 86400.0
ORTHO_TOL = 1e-5         # how far R may stray from a rotation before the calib file is suspect

PARAMS: dict = {
    "velodyne_dir": ("/workspace/data/kitti_raw/2011_09_26/"
                     "2011_09_26_drive_0001_sync/velodyne_points"),
    # Defines t = 0 for the whole replay; must be the SAME drive pipeline_replay's cache was
    # built from, or the cloud is offset from the ego pose by a constant nobody can see.
    "oxts_timestamps": ("/workspace/data/kitti_raw/2011_09_26/"
                        "2011_09_26_drive_0001_extract/oxts/timestamps.txt"),
    "calib_imu_to_velo": "/workspace/data/kitti_raw/2011_09_26/calib_imu_to_velo.txt",
    "stride": 2,
    "frame_id": "velodyne",
}


def _point_field(name: str, offset: int) -> PointField:
    f = PointField()
    f.name = name
    f.offset = offset
    f.datatype = PointField.FLOAT32
    f.count = 1
    return f


# Built once; never mutated, so every message can share them.
FIELDS = [_point_field(name, 4 * i) for i, name in enumerate(("x", "y", "z", "intensity"))]


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


def _load_scan(path: Path, stride: int) -> np.ndarray:
    """(M, 4) float32 [x, y, z, intensity], every `stride`-th point of the raw scan."""
    if path.suffix == ".bin":
        raw = np.fromfile(str(path), dtype=np.float32)
    else:                                    # `_extract` ships the same rows as ASCII
        raw = np.fromfile(str(path), dtype=np.float32, sep=" ")
    if raw.size == 0 or raw.size % 4 != 0:
        raise ValueError(f"{path}: {raw.size} floats is not a whole number of [x,y,z,i] points")
    return np.ascontiguousarray(raw.reshape(-1, 4)[::stride])


def _base_link_to_velodyne(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(R', t') for p_imu = R' p_velo + t', the INVERSE of the file. Docstring has the derivation."""
    rot = trans = None
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key.strip() == "R":
            rot = np.asarray(rest.split(), dtype=float).reshape(3, 3)
        elif key.strip() == "T":
            trans = np.asarray(rest.split(), dtype=float).reshape(3)
    if rot is None or trans is None:
        raise ValueError(f"{path}: expected both an 'R:' and a 'T:' line")
    # A calibration file that is not a rigid transform would silently shear the whole scene.
    if abs(np.abs(rot @ rot.T - np.eye(3)).max()) > ORTHO_TOL:
        raise ValueError(f"{path}: R is not orthonormal within {ORTHO_TOL}")
    if abs(np.linalg.det(rot) - 1.0) > ORTHO_TOL:
        raise ValueError(f"{path}: det(R) = {np.linalg.det(rot):.6f}, expected +1 (a reflection, "
                         f"not a rotation)")
    return rot.T, -(rot.T @ trans)


def _quat_wxyz(m: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (w, x, y, z). Trace branch only, guarded.

    A rigid sensor calibration is within a degree of identity, so w is ~1 and the branch is safe;
    if it is not, the file is not what this node thinks it is and that must be loud, not a
    divide-by-almost-zero that yields a quaternion pointing somewhere arbitrary.
    """
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) * 0.5
    if w < 0.1:
        raise ValueError(f"calibration rotation is {math.degrees(2*math.acos(max(w,0.0))):.1f} deg "
                         f"from identity; expected a near-identity sensor mounting")
    return (w, (m[2, 1] - m[1, 2]) / (4.0 * w),
            (m[0, 2] - m[2, 0]) / (4.0 * w),
            (m[1, 0] - m[0, 1]) / (4.0 * w))


class LidarReplay(Node):
    """Ego-paced KITTI Velodyne publisher. All state is the index of the last scan sent."""

    def __init__(self) -> None:
        super().__init__("lidar_replay")

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = self.get_parameter

        velo_dir = Path(str(p("velodyne_dir").value))
        self._stride = int(p("stride").value)
        self._frame = str(p("frame_id").value)
        if self._stride < 1:
            raise ValueError(f"parameter `stride` must be >= 1, got {self._stride}")

        scan_dir = velo_dir / "data"
        self._scans = sorted(f for f in scan_dir.iterdir() if f.suffix in (".bin", ".txt"))
        if not self._scans:
            raise ValueError(f"{scan_dir} holds no .bin or .txt scans")

        dates, secs = _read_timestamps(velo_dir / "timestamps.txt")
        if len(dates) != len(self._scans):
            raise ValueError(f"{velo_dir}: {len(self._scans)} scan files but {len(dates)} "
                             f"timestamps; the two must index each other 1:1")
        date0, t0 = _split_timestamp(
            Path(str(p("oxts_timestamps").value)).read_text().splitlines()[0])
        # Differencing seconds-of-day across a midnight boundary would be off by 86400 s and look
        # like "the lidar never publishes". drive_0001 is mid-afternoon, so this only ever fires
        # if the node is pointed at unrelated data.
        odd = {d for d in dates if d != date0}
        if odd:
            raise ValueError(f"velodyne timestamps span date(s) {sorted(odd)} but the OXTS time "
                             f"base starts on {date0}; seconds-of-day differencing is invalid")
        self._t = secs - t0                       # replay seconds, the /ego/state time base
        # Nearest-neighbour lookup: searchsorted over the midpoints between consecutive scans.
        self._edges = 0.5 * (self._t[1:] + self._t[:-1])
        self._last = -1

        rot, trans = _base_link_to_velodyne(Path(str(p("calib_imu_to_velo").value)))
        self._static_tf = StaticTransformBroadcaster(self)     # latched; a late viewer still gets it
        self._static_tf.sendTransform(self._base_to_velo(rot, trans))

        # RELIABLE, depth 2. This was BEST_EFFORT to keep a back-pressured websocket from blocking
        # inside the /ego/state callback, and that reasoning is still right -- but BEST_EFFORT does
        # not survive a message this size. A ~950 kB cloud is fragmented into ~700 UDP datagrams,
        # the container's SO_RCVBUF ceiling is net.core.rmem_max = 212992 B (not raisable: it is a
        # non-namespaced sysctl, so no docker --sysctl and no DDS profile can lift it), and
        # BEST_EFFORT has no repair -- one lost fragment discards the whole sample. Measured in
        # isolation, 1 MB samples at 10 Hz: BEST_EFFORT 25/100 delivered, RELIABLE 100/100. In the
        # live view that was ~0.95 Hz of a 2.7 Hz stream reaching Foxglove, i.e. a point cloud that
        # looked absent and an image panel that looked dead.
        #
        # Depth 2 is what makes RELIABLE safe here, and why it must stay shallow: KEEP_LAST(2)
        # bounds the writer history, so a reader that falls behind costs at most `max_blocking_time`
        # (100 ms default) before the oldest sample is DROPPED and publish() returns. The failure
        # mode is therefore unchanged -- drop a frame of a 10 Hz view -- and it is still bounded,
        # while the common case now gets NAK-repaired fragments instead of a coin flip. KEEP_ALL, or
        # a deep history, would reintroduce exactly the unbounded stall the original comment feared.
        cloud_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=2,
                               reliability=QoSReliabilityPolicy.RELIABLE)
        # RELIABLE to match eskf_node's /ego/state publisher -- best-effort here would not match
        # it and the node would receive nothing at all.
        ego_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                             reliability=QoSReliabilityPolicy.RELIABLE)

        self._pub = self.create_publisher(PointCloud2, "/velodyne_points", cloud_qos)
        self.create_subscription(EgoState, "/ego/state", self._on_ego, ego_qos)

        self.get_logger().info(
            f"lidar_replay up: {len(self._scans)} scans from {scan_dir} spanning "
            f"t = {self._t[0]:.3f} .. {self._t[-1]:.3f} s, stride={self._stride}, "
            f"frame={self._frame}; static {BASE_FRAME} -> {self._frame} at "
            f"({trans[0]:+.4f}, {trans[1]:+.4f}, {trans[2]:+.4f}) m")

    # ------------------------------------------------------------------
    # Frames.
    # ------------------------------------------------------------------
    def _base_to_velo(self, rot: np.ndarray, trans: np.ndarray) -> TransformStamped:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = BASE_FRAME        # parent
        tf.child_frame_id = self._frame        # child
        tf.transform.translation.x = float(trans[0])
        tf.transform.translation.y = float(trans[1])
        tf.transform.translation.z = float(trans[2])
        w, x, y, z = _quat_wxyz(rot)
        tf.transform.rotation.w = float(w)
        tf.transform.rotation.x = float(x)
        tf.transform.rotation.y = float(y)
        tf.transform.rotation.z = float(z)
        return tf

    # ------------------------------------------------------------------
    # Pacing.
    # ------------------------------------------------------------------
    def _on_ego(self, msg: EgoState) -> None:
        """Advance to the scan nearest this ego stamp, at most one publish per scan."""
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        index = int(np.searchsorted(self._edges, t))
        if index <= self._last:
            return                      # already sent, or the stream went backwards
        self._last = index
        # A scan recorded BEFORE the replay's t=0 has a negative time on this base, and a ROS
        # stamp is unsigned -- _stamp_from_secs would assert. That happens whenever the OXTS
        # cache starts partway into the drive the Velodyne came from (drive_0091 is the clean
        # tail of drive_0009, so its first 19.15 s of scans predate t=0). Those scans are
        # genuinely outside the replay and must be skipped, not clamped to zero: clamping would
        # pile 19 s of stale geometry onto the first frame.
        if self._t[index] < 0.0:
            return
        self._publish(index)

    def _publish(self, index: int) -> None:
        points = _load_scan(self._scans[index], self._stride)
        msg = PointCloud2()
        # The SCAN's own stamp, on the replay time base -- not the wall clock. Foxglove resolves
        # velodyne -> map through eskf_node's 100 Hz map -> base_link TF at this stamp, so a
        # re-stamped cloud would be drawn against the wrong ego pose.
        msg.header.stamp = _stamp_from_secs(float(self._t[index]))
        msg.header.frame_id = self._frame
        msg.height = 1                                  # unordered
        msg.width = int(points.shape[0])
        msg.fields = FIELDS
        msg.is_bigendian = False
        msg.point_step = POINT_STEP
        msg.row_step = POINT_STEP * msg.width
        msg.is_dense = True                             # KITTI scans carry no NaN/Inf returns
        msg.data = points.tobytes()
        self._pub.publish(msg)
        self.get_logger().info(
            f"scan {index}/{len(self._scans) - 1} at t={self._t[index]:.3f} s, "
            f"{msg.width} points", throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarReplay()
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
