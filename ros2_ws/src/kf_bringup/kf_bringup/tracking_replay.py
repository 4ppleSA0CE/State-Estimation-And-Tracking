"""Synthetic detection replay — drives kf_tracker's TrackerNode and gates parity against Python.

Publishes one DetectionArray per reference frame (empty frames INCLUDED, never skipped, so every
track coasts and ages through the blanked gap), collects /tracks in lock-step, and compares
against data/cache/tracker_py_ref.npz written by scripts/write_py_tracker_refs.py. Mirrors
kitti_replay.py on the localization side.

Gate, per frame: track ids EXACT, track state max-abs error < 1e-6. States are compared as
id->state maps — publication order is deliberately not part of the contract. The 1e-6 bound is
forced by unscented-weight conditioning in the CT mode, not by port quality; see "Why 1e-6, not
1e-9" in docs/superpowers/plans/2026-07-27-stage-5b-kitti-tracker-cpp.md. Do not loosen it to make
a run pass, and do not tighten it.

Exits 0 on PASS and non-zero on any failure (1 = parity/timeout, 2 = unusable reference) so the
launch can be scripted.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from kf_msgs.msg import Detection, DetectionArray, TrackArray
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

DEFAULT_REF_PATH = "/workspace/data/cache/tracker_py_ref.npz"
GEN_CMD = "python3 scripts/write_py_tracker_refs.py"

STATE_TOL = 1e-6      # NOT a parameter, on purpose: the gate is part of the contract, not config
STATE_DIM = 4         # BEV motion state [x, z, vx, vz]
DET_COLS = 7          # det_boxes columns: x, y, z, yaw, l, w, h (score/track_id implicit)
DET_FRAME_ID = "camera"   # KITTI boxes are in camera coords (y down, z forward)
MAX_REPORTED = 5      # cap per-category failure detail so the log stays readable

REQUIRED_KEYS = ("det_boxes", "det_count", "track_ids", "track_states", "track_count")
PROVENANCE_KEYS = ("seed", "n_steps", "git_commit", "git_dirty", "code_sha256")


class ReferenceUnusable(RuntimeError):
    """The reference .npz is absent, unreadable, or malformed — the gate cannot run at all."""


class TrackingReplay(Node):
    def __init__(self) -> None:
        super().__init__("tracking_replay")

        self.declare_parameter("reference_path", DEFAULT_REF_PATH)
        self.declare_parameter("discovery_timeout_s", 30.0)   # wait for tracker_node to appear
        self.declare_parameter("frame_timeout_s", 10.0)       # per-frame wait for one TrackArray
        self.declare_parameter("settle_s", 0.5)               # let DDS finish matching
        self.declare_parameter("drain_s", 1.0)                # catch stragglers / over-publishing

        self._ref_path = Path(self.get_parameter("reference_path").value)
        self._discovery_timeout = float(self.get_parameter("discovery_timeout_s").value)
        self._frame_timeout = float(self.get_parameter("frame_timeout_s").value)
        self._settle = float(self.get_parameter("settle_s").value)
        self._drain = float(self.get_parameter("drain_s").value)

        ref = self._load_reference(self._ref_path)
        self._det_boxes = ref["det_boxes"]        # (F, MAX_DETS, 7), NaN-padded
        self._det_count = ref["det_count"]        # (F,) int32
        self._ref_ids = ref["track_ids"]          # (F, MAX_TRACKS) int32, -1-padded
        self._ref_states = ref["track_states"]    # (F, MAX_TRACKS, 4), NaN-padded
        self._ref_count = ref["track_count"]      # (F,) int32
        self.n_frames = int(self._det_count.shape[0])
        self._validate_reference(ref)
        self._log_provenance(ref)

        # Reliable, deep KeepLast so nothing drops during a fast replay — matches tracker_node.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2000,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._det_pub = self.create_publisher(DetectionArray, "/detections", qos)
        self._track_sub = self.create_subscription(TrackArray, "/tracks", self._on_tracks, qos)

        # One entry per received TrackArray: (ids, states (n, 4)).
        self.received: list[tuple[list[int], np.ndarray]] = []
        self._bad_state_len: list[tuple[int, int, int]] = []   # (frame, track id, len(state))

    # ------------------------------------------------------------------
    # Reference loading / validation / provenance.
    # ------------------------------------------------------------------
    def _load_reference(self, path: Path):
        if not path.is_file():
            raise ReferenceUnusable(
                f"reference .npz not found at {path}\n"
                f"  Generate it on the HOST (it needs the venv's numpy/scipy), from the repo "
                f"root:\n"
                f"      {GEN_CMD}\n"
                f"  then re-run this launch inside the container."
            )
        try:
            ref = np.load(path)
        except Exception as exc:   # noqa: BLE001 — any read failure is equally fatal here
            raise ReferenceUnusable(
                f"could not read {path}: {exc!r}\n  Regenerate it on the HOST with: {GEN_CMD}"
            ) from exc
        missing = [k for k in REQUIRED_KEYS + PROVENANCE_KEYS if k not in ref.files]
        if missing:
            raise ReferenceUnusable(
                f"{path} is missing key(s) {missing} — it predates the current generator.\n"
                f"  Regenerate it on the HOST with: {GEN_CMD}"
            )
        return ref

    def _validate_reference(self, ref) -> None:
        """Fail before publishing anything if the reference cannot support the comparison."""
        f = self.n_frames
        shapes = {
            "det_boxes": (self._det_boxes.ndim == 3 and self._det_boxes.shape[0] == f
                          and self._det_boxes.shape[2] == DET_COLS),
            "track_ids": (self._ref_ids.ndim == 2 and self._ref_ids.shape[0] == f),
            "track_states": (self._ref_states.ndim == 3 and self._ref_states.shape[0] == f
                             and self._ref_states.shape[2] == STATE_DIM),
            "track_count": (self._ref_count.ndim == 1 and self._ref_count.shape[0] == f),
        }
        bad = [k for k, ok in shapes.items() if not ok]
        if bad:
            raise ReferenceUnusable(
                f"{self._ref_path}: unexpected shape for {bad} "
                f"(det_boxes={self._det_boxes.shape}, track_ids={self._ref_ids.shape}, "
                f"track_states={self._ref_states.shape}, track_count={self._ref_count.shape}, "
                f"frames={f}); expected (F, *, {DET_COLS}) / (F, *) / (F, *, {STATE_DIM}) / (F,)."
            )
        if f == 0:
            raise ReferenceUnusable(f"{self._ref_path}: zero frames — nothing to replay.")
        if self._ref_ids.shape[1] != self._ref_states.shape[1]:
            raise ReferenceUnusable(
                f"{self._ref_path}: track_ids and track_states disagree on capacity "
                f"({self._ref_ids.shape[1]} vs {self._ref_states.shape[1]})."
            )

        for k in range(f):
            n_det = int(self._det_count[k])
            if not (0 <= n_det <= self._det_boxes.shape[1]):
                raise ReferenceUnusable(
                    f"{self._ref_path}: frame {k} det_count={n_det} outside "
                    f"[0, {self._det_boxes.shape[1]}]."
                )
            if not np.isfinite(self._det_boxes[k, :n_det]).all():
                raise ReferenceUnusable(
                    f"{self._ref_path}: frame {k} has a non-finite value inside the first "
                    f"{n_det} detection rows (padding beyond det_count is NaN by design)."
                )
            n_trk = int(self._ref_count[k])
            if not (0 <= n_trk <= self._ref_ids.shape[1]):
                raise ReferenceUnusable(
                    f"{self._ref_path}: frame {k} track_count={n_trk} outside "
                    f"[0, {self._ref_ids.shape[1]}]."
                )
            if not np.isfinite(self._ref_states[k, :n_trk]).all():
                raise ReferenceUnusable(
                    f"{self._ref_path}: frame {k} has a non-finite value inside the first "
                    f"{n_trk} track states."
                )
            ids_k = self._ref_ids[k, :n_trk]
            if n_trk and (ids_k < 0).any():
                raise ReferenceUnusable(
                    f"{self._ref_path}: frame {k} has a negative id inside track_count={n_trk} "
                    f"({ids_k.tolist()}); -1 is the padding sentinel."
                )
            if n_trk and len(set(int(v) for v in ids_k)) != n_trk:
                raise ReferenceUnusable(
                    f"{self._ref_path}: frame {k} repeats a track id ({ids_k.tolist()}); the "
                    f"id->state map comparison would silently collapse it."
                )

        n_steps = int(ref["n_steps"].item())
        if n_steps != f:
            self.get_logger().warn(
                f"reference n_steps={n_steps} but det_count has {f} frames; replaying {f}."
            )

    def _log_provenance(self, ref) -> None:
        """Log every provenance field so a stale or foreign reference is visible, not silent.

        Regenerating against a different working tree was measured during Task 9 to move tracker
        state by ~100 m and flip ids in about half the frames (see
        scripts/write_py_tracker_refs.py::_git_provenance for that measurement, not repeated
        here), so these lines are the first thing to compare when a passing gate starts failing.
        """
        empty = np.flatnonzero(self._det_count == 0).tolist()
        blank = (ref["blank_frame_indices"].tolist()
                 if "blank_frame_indices" in ref.files else "n/a")
        distinct = len({int(v) for v in self._ref_ids.ravel() if v >= 0})
        log = self.get_logger()
        log.info(f"reference: {self._ref_path}")
        log.info(
            f"provenance: seed={int(ref['seed'].item())} n_steps={int(ref['n_steps'].item())} "
            f"git_commit={ref['git_commit'].item()} git_dirty={bool(ref['git_dirty'].item())}"
        )
        log.info(f"provenance: code_sha256={ref['code_sha256'].item()}")
        log.info(
            f"scenario: frames={self.n_frames} total_dets={int(self._det_count.sum())} "
            f"total_tracks={int(self._ref_count.sum())} distinct_ids={distinct} "
            f"blank_frame_indices={blank} empty_det_frames={empty} "
            f"(empty frames are PUBLISHED, not skipped, so tracks coast and age)"
        )
        log.info(f"gate: state max-abs < {STATE_TOL:.0e}, ids exact, per frame")

    # ------------------------------------------------------------------
    # Replay.
    # ------------------------------------------------------------------
    def _detection_msg(self, k: int) -> DetectionArray:
        msg = DetectionArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = DET_FRAME_ID
        for j in range(int(self._det_count[k])):
            x, y, z, yaw, length, width, height = (float(v) for v in self._det_boxes[k, j])
            d = Detection()
            d.pose.position.x = x
            d.pose.position.y = y
            d.pose.position.z = z
            # Yaw about the vertical axis. With x = z = 0 the node's decode
            # atan2(2(wy + xz), 1 - 2(y^2 + x^2)) = atan2(sin yaw, cos yaw) returns this yaw back.
            d.pose.orientation.w = math.cos(yaw * 0.5)
            d.pose.orientation.x = 0.0
            d.pose.orientation.y = math.sin(yaw * 0.5)
            d.pose.orientation.z = 0.0
            d.dimensions.x = length
            d.dimensions.y = width
            d.dimensions.z = height
            d.object_id = -1     # anonymous detection; score is implicit (Box3D defaults to 1.0)
            msg.detections.append(d)
        return msg

    def _on_tracks(self, msg: TrackArray) -> None:
        """Snapshot ids + state NOW (rclpy hands us a fresh message, but keep the habit — the
        C++ side's BoxTrack/ImmFilter mutate in place, which is what this reference guards)."""
        frame = len(self.received)
        ids: list[int] = []
        states: list[np.ndarray] = []
        for t in msg.tracks:
            tid = int(t.id)
            s = np.asarray(t.state, dtype=float)
            if s.shape != (STATE_DIM,):
                self._bad_state_len.append((frame, tid, int(s.size)))
                s = np.full(STATE_DIM, np.nan)
            ids.append(tid)
            states.append(s)
        arr = (np.stack(states) if states
               else np.zeros((0, STATE_DIM), dtype=float))
        self.received.append((ids, arr))

    def _await_peers(self) -> bool:
        """Wait until tracker_node has matched us in both directions before the first publish —
        RELIABLE+VOLATILE does not redeliver to a late-joining peer, so publishing early silently
        loses frames."""
        deadline = time.monotonic() + self._discovery_timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._det_pub.get_subscription_count() > 0 and self.count_publishers("/tracks") > 0:
                time.sleep(self._settle)
                self.get_logger().info("tracker_node discovered on /detections and /tracks")
                return True
        self.get_logger().error(
            f"FAIL(no-peer): no node matched /detections (subs="
            f"{self._det_pub.get_subscription_count()}) and /tracks (pubs="
            f"{self.count_publishers('/tracks')}) within {self._discovery_timeout:.1f} s. "
            f"Is kf_tracker's tracker_node running? Check its log for a startup crash."
        )
        return False

    def _await_frame(self, k: int) -> bool:
        deadline = time.monotonic() + self._frame_timeout
        while len(self.received) < k + 1 and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if len(self.received) >= k + 1:
            return True
        detail = ("the /tracks subscription never fired at all"
                  if not self.received else
                  f"{len(self.received)} of {self.n_frames} frames arrived before the stall")
        self.get_logger().error(
            f"FAIL(timeout): frame {k} produced no /tracks within {self._frame_timeout:.1f} s — "
            f"{detail}. The tracker node stopped publishing: check its log for a crash "
            f"(a sigma-point/assignment throw escapes an rclcpp callback and kills the process)."
        )
        return False

    def run(self) -> int:
        if not self._await_peers():
            return 1

        for k in range(self.n_frames):
            self._det_pub.publish(self._detection_msg(k))   # empty frames included, on purpose
            if not self._await_frame(k):
                return 1

        # Keep spinning briefly so a node that publishes MORE than one TrackArray per frame is
        # caught by the frame-count check rather than passing silently.
        deadline = time.monotonic() + self._drain
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

        return self._check_parity()

    # ------------------------------------------------------------------
    # Parity gate.
    # ------------------------------------------------------------------
    def _check_parity(self) -> int:
        log = self.get_logger()
        n_recv = len(self.received)
        n = min(n_recv, self.n_frames)

        count_bad: list[str] = []     # frame produced the wrong number of tracks
        id_bad: list[str] = []        # right count, wrong id set (or a duplicate id)
        state_bad: list[str] = []     # ids agree, state error >= STATE_TOL
        worst_err = 0.0
        worst_where = "n/a"

        for k in range(n):
            ids, states = self.received[k]
            n_ref = int(self._ref_count[k])
            if len(ids) != n_ref:
                want_ids = sorted(int(v) for v in self._ref_ids[k, :n_ref])
                count_bad.append(f"frame {k}: got {len(ids)} tracks {sorted(ids)}, "
                                 f"expected {n_ref} {want_ids}")
                continue
            got = {tid: states[i] for i, tid in enumerate(ids)}
            if len(got) != len(ids):
                id_bad.append(f"frame {k}: duplicate track id in {sorted(ids)}")
                continue
            want = {int(self._ref_ids[k, j]): self._ref_states[k, j] for j in range(n_ref)}
            if set(got) != set(want):
                id_bad.append(f"frame {k}: ids {sorted(got)} != expected {sorted(want)} "
                              f"(missing {sorted(set(want) - set(got))}, "
                              f"extra {sorted(set(got) - set(want))})")
                continue
            if n_ref:
                order = sorted(want)
                # np.max/np.argmax PROPAGATE NaN (argmax returns the first NaN's index), so a
                # diverged track that publishes NaN lands here as the frame's worst error instead
                # of being skipped by a plain `err > worst` comparison.
                per_track = np.max(np.abs(np.stack([got[t] for t in order])
                                          - np.stack([want[t] for t in order])), axis=1)
                idx = int(np.argmax(per_track))
                frame_err, frame_tid = float(per_track[idx]), order[idx]
            else:
                frame_err, frame_tid = 0.0, -1
            if not math.isnan(worst_err) and (math.isnan(frame_err) or frame_err > worst_err):
                worst_err, worst_where = frame_err, f"frame {k} track {frame_tid}"
            if not (frame_err < STATE_TOL):     # NaN-safe: a non-finite state fails here
                note = " (non-finite state published)" if not math.isfinite(frame_err) else ""
                state_bad.append(f"frame {k}: track {frame_tid} state max-abs error "
                                 f"{frame_err:.6e} >= {STATE_TOL:.0e}{note}")

        frames_ok = n_recv == self.n_frames
        ok = (frames_ok and not count_bad and not id_bad and not state_bad
              and not self._bad_state_len)

        log.info(
            f"PARITY {'PASS' if ok else 'FAIL'}: frames={n_recv}/{self.n_frames} "
            f"state_max_abs_err={worst_err:.3e} (tol {STATE_TOL:.0e}, worst at {worst_where}) "
            f"id_mismatches={len(id_bad)} count_mismatches={len(count_bad)} "
            f"state_mismatches={len(state_bad)}"
        )
        if ok:
            return 0

        if not frames_ok:
            if n_recv == 0:
                log.error("FAIL(frames): no TrackArray was ever received — the /tracks "
                          "subscription never fired.")
            else:
                why = ("the node published MORE than one TrackArray per DetectionArray"
                       if n_recv > self.n_frames else
                       "the node stopped publishing before the last frame")
                log.error(
                    f"FAIL(frames): received {n_recv} TrackArray messages, expected "
                    f"{self.n_frames} (one per published DetectionArray, empty frames "
                    f"included) — {why}."
                )
        for label, entries in (("counts", count_bad), ("ids", id_bad), ("state", state_bad)):
            if not entries:
                continue
            log.error(f"FAIL({label}): {len(entries)} frame(s); first "
                      f"{min(MAX_REPORTED, len(entries))}:")
            for line in entries[:MAX_REPORTED]:
                log.error(f"  {line}")
        if self._bad_state_len:
            log.error(
                f"FAIL(schema): {len(self._bad_state_len)} track(s) published a state of the "
                f"wrong length (expected {STATE_DIM}); first: {self._bad_state_len[:MAX_REPORTED]}"
            )
        if state_bad and not id_bad and not count_bad and not self._bad_state_len:
            log.error(
                "state-only divergence: do NOT loosen STATE_TOL. See 'Why 1e-6, not 1e-9' in "
                "docs/superpowers/plans/2026-07-27-stage-5b-kitti-tracker-cpp.md for the "
                "suspect list, and re-check the provenance lines above against a freshly "
                f"generated reference ({GEN_CMD})."
            )
        return 1


def main() -> None:
    rclpy.init()
    node = None
    code = 1
    try:
        node = TrackingReplay()
        code = node.run()
    except ReferenceUnusable as exc:
        print(f"PARITY FAIL(reference): {exc}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("PARITY FAIL: interrupted before the gate ran", file=sys.stderr)
        code = 130
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
