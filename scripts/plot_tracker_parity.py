"""Record the C++ tracker_node's /tracks and plot them against the Python reference.

Produces output/tracker_cpp_parity.png: a BEV view where the Python reference tracks are drawn as
lines and the C++ node's published tracks as scattered points on top. If the port is faithful the
points sit on the lines and the per-frame error curve stays under the 1e-6 gate.

Run INSIDE the ROS2 container, with the tracker_node already reachable:
    ros2 run kf_tracker tracker_node --ros-args --params-file <tracker.yaml> &
    python3 scripts/plot_tracker_parity.py

Or let it drive everything itself (default): it publishes the reference detections the same way
tracking_replay.py does, records the replies, and writes the figure.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "data" / "cache" / "tracker_py_ref.npz"
OUT = ROOT / "output" / "tracker_cpp_parity.png"
STATE_TOL = 1e-6


def _record() -> tuple[list[list[tuple[int, np.ndarray]]], dict]:
    """Publish the reference detections, collect /tracks. Returns (per-frame [(id, state)], ref)."""
    import rclpy
    from kf_msgs.msg import Detection, DetectionArray, TrackArray
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

    ref = np.load(REF)
    det_boxes, det_count = ref["det_boxes"], ref["det_count"]
    n_frames = int(det_count.shape[0])

    rclpy.init()
    node = Node("tracker_parity_plotter")
    qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=2000,
                     reliability=QoSReliabilityPolicy.RELIABLE)
    pub = node.create_publisher(DetectionArray, "/detections", qos)
    got: list[list[tuple[int, np.ndarray]]] = []
    node.create_subscription(
        TrackArray, "/tracks",
        lambda m: got.append([(int(t.id), np.asarray(t.state, dtype=float)) for t in m.tracks]),
        qos)

    # wait for the node to appear on both topics, else RELIABLE+VOLATILE drops the early frames
    deadline = node.get_clock().now().nanoseconds + 30_000_000_000
    while node.get_clock().now().nanoseconds < deadline:
        if pub.get_subscription_count() > 0 and node.count_publishers("/tracks") > 0:
            break
        rclpy.spin_once(node, timeout_sec=0.05)
    else:
        raise SystemExit("tracker_node never appeared on /detections and /tracks")
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.02)

    for k in range(n_frames):
        msg = DetectionArray()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        for j in range(int(det_count[k])):
            x, y, z, yaw, l, w, h = det_boxes[k, j]
            d = Detection()
            d.pose.position.x, d.pose.position.y, d.pose.position.z = float(x), float(y), float(z)
            d.pose.orientation.w = float(math.cos(yaw * 0.5))
            d.pose.orientation.y = float(math.sin(yaw * 0.5))
            d.dimensions.x, d.dimensions.y, d.dimensions.z = float(l), float(w), float(h)
            d.object_id = -1
            msg.detections.append(d)
        pub.publish(msg)
        target = len(got) + 1
        end = node.get_clock().now().nanoseconds + 5_000_000_000
        while len(got) < target and node.get_clock().now().nanoseconds < end:
            rclpy.spin_once(node, timeout_sec=0.02)

    node.destroy_node()
    rclpy.shutdown()
    return got, ref


def _plot(got, ref) -> None:
    import matplotlib.pyplot as plt

    ids, states, counts = ref["track_ids"], ref["track_states"], ref["track_count"]
    n = len(got)

    # per-frame max-abs error, comparing id->state maps
    errs = np.zeros(n)
    for k in range(n):
        m = int(counts[k])
        want = {int(ids[k, j]): states[k, j] for j in range(m)}
        e = 0.0
        for tid, st in got[k]:
            if tid in want:
                e = max(e, float(np.max(np.abs(st - want[tid]))))
        errs[k] = e

    # python reference trajectories per track id
    tracks: dict[int, list[tuple[float, float]]] = {}
    for k in range(n):
        for j in range(int(counts[k])):
            tracks.setdefault(int(ids[k, j]), []).append((states[k, j, 0], states[k, j, 1]))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                  gridspec_kw={"height_ratios": [3, 1]})
    for tid, pts in tracks.items():
        a = np.asarray(pts)
        ax.plot(a[:, 0], a[:, 1], "-", color="0.55", linewidth=2.5,
                zorder=1, solid_capstyle="round")
    for k in range(n):
        for tid, st in got[k]:
            ax.scatter(st[0], st[1], s=9, c=f"C{tid % 10}", zorder=2, linewidths=0)
    ax.plot([], [], "-", color="0.55", linewidth=2.5, label="Python reference (line)")
    ax.scatter([], [], s=20, c="C0", label="C++ tracker_node /tracks (points)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.set_title(f"Stage 5B parity — C++ tracker_node vs Python reference\n"
                 f"{n} frames, max-abs state error {errs.max():.3e} (gate {STATE_TOL:.0e})")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3); ax.set_aspect("equal", adjustable="datalim")

    ax2.semilogy(np.arange(n), np.maximum(errs, 1e-18), lw=1.2, color="C3")
    ax2.axhline(STATE_TOL, ls="--", color="k", lw=1, label=f"gate {STATE_TOL:.0e}")
    ax2.set_xlabel("frame"); ax2.set_ylabel("max |Δstate|")
    ax2.grid(True, alpha=0.3, which="both"); ax2.legend(loc="best")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"frames={n} max_abs_err={errs.max():.6e} gate={STATE_TOL:.0e} "
          f"-> {'PASS' if errs.max() < STATE_TOL else 'FAIL'}")
    print(f"saved {OUT}")


DUMP = ROOT / "data" / "cache" / "tracker_cpp_tracks.npz"
MAXT = 16


def _save(got) -> None:
    """Pad the recorded C++ tracks into rectangular arrays so the host can plot them.

    The container has rclpy but no matplotlib; the host has matplotlib but no rclpy. So `--record`
    runs inside the container and `--plot` runs on the host.
    """
    n = len(got)
    ids = np.full((n, MAXT), -1, dtype=np.int32)
    st = np.full((n, MAXT, 4), np.nan)
    cnt = np.zeros(n, dtype=np.int32)
    for k, rows in enumerate(got):
        if len(rows) > MAXT:
            raise SystemExit(f"frame {k}: {len(rows)} tracks exceeds MAXT={MAXT}")
        cnt[k] = len(rows)
        for j, (tid, s) in enumerate(rows):
            ids[k, j] = tid
            st[k, j] = s
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DUMP, track_ids=ids, track_states=st, track_count=cnt)
    print(f"recorded {n} frames, {int(cnt.sum())} track rows -> {DUMP}")


def _load():
    d = np.load(DUMP)
    ids, st, cnt = d["track_ids"], d["track_states"], d["track_count"]
    return [[(int(ids[k, j]), st[k, j]) for j in range(int(cnt[k]))] for k in range(len(cnt))]


if __name__ == "__main__":
    if not REF.exists():
        raise SystemExit(f"reference not found at {REF}; run scripts/write_py_tracker_refs.py first")
    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"
    if mode == "--record":                      # in the container (needs rclpy)
        got, _ = _record()
        if not got:
            raise SystemExit("no /tracks received - is tracker_node running?")
        _save(got)
    elif mode == "--plot":                      # on the host (needs matplotlib)
        _plot(_load(), np.load(REF))
    else:
        got, ref = _record()
        if not got:
            raise SystemExit("no /tracks received - is tracker_node running?")
        _plot(got, ref)
    sys.exit(0)
