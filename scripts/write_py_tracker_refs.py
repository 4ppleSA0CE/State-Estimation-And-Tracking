"""Write the Python IMM-tracker reference (detections + expected tracks) for the C++ parity gate.

The C++ tracker node (kf_tracker) is validated against this reference by
ros2_ws/src/kf_bringup/kf_bringup/tracking_replay.py::_check_parity, mirroring how
scripts/write_py_refs.py + kitti_replay.py gate the ESKF.

The scenario is built here rather than in tracking/scenario_sim.py so no Stage 4 module changes:
scenario_sim emits 2-D points, and this wraps them into Box3D detections with fixed car geometry.

Validation runs on the exact cost matrices associate_from_cost solves, recorded in the same pass
that writes the reference (not a re-derived stand-in): every optimal assignment must be provably
unique (forbid-and-resolve, not diffing cost entries) and every IoU must clear iou_gate with
margin. Two consecutive frames are forced empty so step([]) -- predict/coast/age with no
detections -- is actually exercised. Seed, step count, git HEAD/dirty state, and a sha256 of the
actual prototypes/python source used are stamped into the .npz so a stale or foreign reference is
visible rather than silently trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prototypes" / "python"))
sys.path.insert(0, str(ROOT / "prototypes" / "python" / "tracking"))

import kitti_tracker  # noqa: E402  module handle so iou_3d can be monkeypatched (not edited)
from association import BIG_COST  # noqa: E402
from kitti_boxes import Box3D  # noqa: E402
from kitti_tracker import KittiTracker, KittiTrackerConfig  # noqa: E402
from scenario_sim import SimConfig, simulate  # noqa: E402

CAR_L, CAR_W, CAR_H, CAR_Y = 3.9, 1.6, 1.5, 1.6
MAX_DETS_PER_FRAME = 16
MAX_TRACKS_PER_FRAME = 16

# Force these two consecutive frames to zero detections so the reference actually exercises
# step([]): predict/coast/age every track with no association at all. Natural "all 4 targets
# missed AND Poisson(2) clutter == 0" coincidence is ~1.35e-5/frame, so no reachable seed produces
# this on its own (0 of 30,000 frames checked across seeds 0..299). Two in a row exercises
# coast-twice under max_age=2 without necessarily killing every track. Keep these -- do not
# "clean them up".
BLANK_FRAME_INDICES = (40, 41)

AMBIGUITY_EPS = 1e-9      # assignment margin below this is a genuine tie, not float noise
GATE_MARGIN_EPS = 1e-9    # |iou - iou_gate| below this is not reliably reproducible


def _boxes_for_frame(points) -> list[Box3D]:
    """Wrap 2-D (x, y) sim detections as BEV (x, z) car boxes with fixed geometry."""
    return [Box3D(float(p[0]), CAR_Y, float(p[1]), 0.0, CAR_L, CAR_W, CAR_H) for p in points]


def _git_provenance(root: Path) -> tuple[str, bool]:
    """(commit_hash, dirty). The reference numerics depend on the exact working tree -- e.g. an
    uncommitted fix to prototypes/python/imm_synthetic.py currently in this checkout changes
    tracker state by ~100 m and flips ids in about half the frames if it's reverted -- so a stale
    or foreign reference must be visible in the artifact, not silent. Degrades to "unknown" /
    dirty=True if git is unavailable.

    This is repo-wide, coarse provenance only: git_dirty trips on any uncommitted change anywhere
    in the repo (docs/, ros2_ws/, ...) whether or not it touches code this reference actually
    runs, and it cannot distinguish "committed, then reverted the fix" from "committed the fix"
    since both stamp the same commit hash. See _code_sha256 for the content-addressed check that
    actually answers "did the code this reference depends on change"."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown", True
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        dirty = bool(status.strip())
    except (subprocess.SubprocessError, OSError):
        dirty = True
    return commit, dirty


def _code_sha256(root: Path) -> str:
    """sha256 over the source of every already-imported module that resolves under
    prototypes/python, plus this script -- keyed and hashed in module-name sorted order so the
    result is deterministic regardless of import order.

    git_commit/git_dirty (see _git_provenance) stamp repo state, not the code this reference
    actually depends on: they cannot tell "imm_synthetic.py has an uncommitted fix" apart from
    "docs/ has an uncommitted typo", and they stamp identically whether that fix is present or
    reverted. This hash instead moves if and only if a module actually imported into this run
    changes, independent of git state."""
    proto_root = (root / "prototypes" / "python").resolve()
    sources: dict[str, Path] = {}
    for name, mod in list(sys.modules.items()):
        file = getattr(mod, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        try:
            path.relative_to(proto_root)
        except ValueError:
            continue    # not a prototypes/python module (stdlib, numpy, scipy, ...) -- skip
        sources[name] = path
    sources["<this script>"] = Path(__file__).resolve()   # lives under scripts/, add explicitly

    digest = hashlib.sha256()
    for name in sorted(sources):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sources[name].read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class _RecordingTracker(KittiTracker):
    """KittiTracker with _cost overridden to copy out every matrix associate_from_cost actually
    solves. This changes nothing about what gets computed -- it calls the real _cost and records
    its return value -- so it's safe to use for the reference-generating pass itself: validation
    then checks exactly what produced this artifact, not a stale or re-derived stand-in. (The
    previous guard called predicted_box() before step() had called predict(), i.e. it inspected
    the wrong frame's priors entirely.)"""

    def __init__(self, cfg: KittiTrackerConfig) -> None:
        super().__init__(cfg)
        self.recorded: list[tuple[int, np.ndarray]] = []
        self.frame_idx = -1   # caller sets this immediately before each step()

    def _cost(self, dets: list[Box3D]) -> np.ndarray:
        cost = super()._cost(dets)
        self.recorded.append((self.frame_idx, cost.copy()))
        return cost


@contextmanager
def _record_ious(tracker_module):
    """Monkeypatch tracker_module.iou_3d to record every raw IoU it computes, gated-out pairs
    included (those never appear in a cost matrix, only as BIG_COST), so the closest approach to
    the discrete iou_gate boundary can be measured. Restores the original function on exit."""
    ious: list[float] = []
    real_iou_3d = tracker_module.iou_3d

    def _wrapped(a, b):
        val = real_iou_3d(a, b)
        ious.append(val)
        return val

    tracker_module.iou_3d = _wrapped
    try:
        yield ious
    finally:
        tracker_module.iou_3d = real_iou_3d


def _check_assignment_uniqueness(cost: np.ndarray) -> float:
    """Equality of two cost entries is neither necessary nor sufficient for a unique optimal
    assignment, so test the assignment directly: solve, then for every matched cell (cost <
    BIG_COST) forbid it and re-solve, scoring the alternative on the ORIGINAL matrix. The margin
    is how much worse the best alternative is; ~0 means a second, equally-optimal assignment
    exists that disagrees with this one.

    Scored via the symmetric difference of assignment cells, summed exactly with math.fsum,
    rather than (alt_total - base_total) from two independent cost.sum() calls: with BIG_COST
    cells included those sums reach ~7e6, where 1 ULP is ~9.3e-10 -- larger than AMBIGUITY_EPS --
    so the subtraction's own rounding error could mask a genuine tie as a spuriously positive
    margin. Cells shared by both assignments are excluded from the symmetric difference, i.e.
    they cancel by construction here rather than via float subtraction of two large sums."""
    n, m = cost.shape
    if n == 1 and m == 1:
        return float("inf")   # exactly one assignment can exist at all; nothing to compare
    rows, cols = linear_sum_assignment(cost)
    bse = set(zip(map(int, rows), map(int, cols)))
    # Must dominate the worst case of an assignment built entirely from gated-out cells, or the
    # solver could just keep the "forbidden" cell because avoiding it looks pricier than it is.
    forbid = BIG_COST * (n + m) * 10.0
    min_margin = float("inf")
    for i, j in zip(rows, cols):
        if cost[i, j] >= BIG_COST:
            continue
        forced = cost.copy()
        forced[i, j] = forbid
        alt_rows, alt_cols = linear_sum_assignment(forced)
        alt = set(zip(map(int, alt_rows), map(int, alt_cols)))
        margin = math.fsum([float(cost[a, b]) for a, b in alt - bse]
                           + [-float(cost[a, b]) for a, b in bse - alt])
        min_margin = min(min_margin, margin)
    return min_margin


def _check_gap_survival(track_ids: np.ndarray, blank_indices: tuple[int, ...]) -> int | None:
    """A track id confirmed both immediately before and immediately after the blanked gap proves
    it coasted through and was re-associated afterwards. None means the gap killed every track.

    blank_indices[0]-1 / blank_indices[-1]+1 only bracket a single coasted gap if blank_indices is
    contiguous -- a non-contiguous set (e.g. frames 40 and 60) would let a track that was merely
    re-associated in between look like it "survived" a gap it never coasted through at all."""
    expected = tuple(range(min(blank_indices), max(blank_indices) + 1))
    if tuple(blank_indices) != expected:
        raise SystemExit(
            f"blank_indices={blank_indices} are not contiguous (expected {expected}) -- "
            f"before/after of a non-contiguous set does not bracket a single coasted gap."
        )
    before = {int(i) for i in track_ids[blank_indices[0] - 1] if i >= 0}
    after = {int(i) for i in track_ids[blank_indices[-1] + 1] if i >= 0}
    survivors = before & after
    return min(survivors) if survivors else None


def _run_pass(
    sim_cfg: SimConfig,
    seed: int,
    blank_indices: tuple[int, ...],
    tracker_cfg: KittiTrackerConfig,
) -> dict:
    """Run one full simulate-and-track pass with the given frames forced to zero detections.

    Raises SystemExit for the correctness checks (assignment ambiguity, IoU-gate margin, and
    per-frame count overflow): those mean the reference is unusable outright, regardless of which
    frames got blanked. Gap survival is deliberately NOT raised here -- it comes back as
    survivor_id=None so the caller (main()) can retry with a different blank_indices before
    deciding the scenario is unusable for this seed.

    scenario_sim.simulate() constructs its own np.random.default_rng(seed) fresh on every call, so
    calling this twice with the same (sim_cfg, seed) and different blank_indices tracks the
    identical underlying detection stream -- only the forced-blank frames differ."""
    point_frames, _gt = simulate(sim_cfg, seed=seed)
    frames = [_boxes_for_frame(f) for f in point_frames]
    for idx in blank_indices:                   # force the empty-detection frame(s), see comment
        frames[idx] = []                        # above BLANK_FRAME_INDICES

    # Fixed-width padded arrays so the .npz stays rectangular; counts say how many rows are real.
    n = len(frames)
    det_xyz = np.full((n, MAX_DETS_PER_FRAME, 7), np.nan)   # x, y, z, yaw, l, w, h
    det_count = np.zeros(n, dtype=np.int32)
    trk_id = np.full((n, MAX_TRACKS_PER_FRAME), -1, dtype=np.int32)
    trk_state = np.full((n, MAX_TRACKS_PER_FRAME, 4), np.nan)   # x, z, vx, vz
    trk_count = np.zeros(n, dtype=np.int32)

    # Single pass: the same tracker instance both produces the reference AND records every cost
    # matrix associate_from_cost actually solves, so validation below checks exactly what
    # produced this artifact -- never a decorrelated re-derivation.
    trk = _RecordingTracker(tracker_cfg)
    with _record_ious(kitti_tracker) as ious:
        for k, boxes in enumerate(frames):
            if len(boxes) > MAX_DETS_PER_FRAME:
                raise SystemExit(f"frame {k}: {len(boxes)} detections exceeds MAX_DETS_PER_FRAME")
            det_count[k] = len(boxes)
            for j, b in enumerate(boxes):
                det_xyz[k, j] = [b.x, b.y, b.z, b.yaw, b.l, b.w, b.h]

            trk.frame_idx = k
            confirmed = trk.step(boxes)
            if len(confirmed) > MAX_TRACKS_PER_FRAME:
                raise SystemExit(
                    f"frame {k}: {len(confirmed)} tracks exceeds MAX_TRACKS_PER_FRAME"
                )
            trk_count[k] = len(confirmed)
            # Snapshot NOW -- Track/IMM state mutates in place, so a deferred read returns the
            # FINAL state for every frame (the Stage 4 bug; see docs/notes/tracking_imm_writeup.md).
            for j, t in enumerate(confirmed):
                x, _p = t.imm.state()
                trk_id[k, j] = t.id
                trk_state[k, j] = [x[0], x[1], x[2], x[3]]

    # Optimum uniqueness, tested directly on the real matrices -- not by diffing cost entries.
    if not trk.recorded:
        raise SystemExit("no non-trivial cost matrix was ever solved; nothing to validate")
    min_assignment_margin = float("inf")
    for frame_idx, cost in trk.recorded:
        margin = _check_assignment_uniqueness(cost)
        min_assignment_margin = min(min_assignment_margin, margin)
        if margin < AMBIGUITY_EPS:
            raise SystemExit(
                f"frame {frame_idx}: assignment margin {margin!r} < {AMBIGUITY_EPS} -- an "
                f"alternate optimal assignment exists; the reference is unusable as an id gate."
            )

    # The discrete iou >= iou_gate test must not be sitting on a knife edge either.
    if not ious:
        raise SystemExit("no IoU pairs were ever evaluated -- scenario or gate is degenerate")
    min_gate_margin = min(abs(v - tracker_cfg.iou_gate) for v in ious)
    if min_gate_margin < GATE_MARGIN_EPS:
        raise SystemExit(
            f"min |iou - iou_gate| = {min_gate_margin!r} < {GATE_MARGIN_EPS} -- a real IoU sits "
            f"within reproduction error of the gate boundary and could flip cost <-> BIG_COST."
        )

    # Confirm the forced empty frame(s) actually exercised coast-and-recover, not coast-to-death.
    # None means the gap killed every track -- left to the caller to retry or give up on (not
    # raised here; see docstring).
    survivor_id = _check_gap_survival(trk_id, blank_indices)

    return {
        "det_xyz": det_xyz,
        "det_count": det_count,
        "trk_id": trk_id,
        "trk_state": trk_state,
        "trk_count": trk_count,
        "min_assignment_margin": min_assignment_margin,
        "min_gate_margin": min_gate_margin,
        "survivor_id": survivor_id,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=100)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if min(BLANK_FRAME_INDICES) < 1 or max(BLANK_FRAME_INDICES) > args.n_steps - 2:
        raise SystemExit(
            f"BLANK_FRAME_INDICES={BLANK_FRAME_INDICES} need >=1 frame of margin on both sides "
            f"of --n-steps={args.n_steps} for the gap-survival check; adjust one or the other."
        )

    sim_cfg = SimConfig(n_steps=args.n_steps)
    tracker_cfg = KittiTrackerConfig()
    if tracker_cfg.cost != "iou":   # the validation below only understands the iou path
        raise SystemExit(
            f"reference validation only checks the 'iou' cost path, got {tracker_cfg.cost!r}"
        )

    # Two blanked frames exercise coast-twice under max_age=2 (see BLANK_FRAME_INDICES comment
    # above), but the scenario is random per --seed: a handful of seeds (2, 7, 18 of 0..19) happen
    # to leave zero tracks alive across a 2-frame gap. That is a usability limit on --seed, not a
    # correctness problem with the reference, so retry once with a single blanked frame instead of
    # aborting outright; --seed 0 (the default) never needs the fallback.
    result = _run_pass(sim_cfg, args.seed, BLANK_FRAME_INDICES, tracker_cfg)
    blank_used = BLANK_FRAME_INDICES
    if result["survivor_id"] is None:
        single = (BLANK_FRAME_INDICES[0],)
        print(
            f"gap survival: two-frame gap {BLANK_FRAME_INDICES} left no track id confirmed on "
            f"both sides for seed={args.seed}; retrying with single-frame gap {single}"
        )
        result = _run_pass(sim_cfg, args.seed, single, tracker_cfg)
        blank_used = single
        if result["survivor_id"] is None:
            raise SystemExit(
                f"no confirmed track survived the blanked gap at frames {BLANK_FRAME_INDICES} "
                f"or the single-frame fallback {single} for seed={args.seed} -- try a different "
                f"--seed."
            )
        print(f"gap survival variant used: SINGLE blanked frame {blank_used} "
              f"(two-frame gap {BLANK_FRAME_INDICES} failed for this seed)")
    else:
        print(f"gap survival variant used: TWO blanked frames {blank_used}")

    commit, dirty = _git_provenance(ROOT)
    code_sha256 = _code_sha256(ROOT)

    det_count, trk_id, trk_count = result["det_count"], result["trk_id"], result["trk_count"]

    out = args.out_dir / "tracker_py_ref.npz"
    np.savez(
        out,
        det_boxes=result["det_xyz"],
        det_count=det_count,
        track_ids=trk_id,
        track_states=result["trk_state"],
        track_count=trk_count,
        seed=np.array(args.seed),
        n_steps=np.array(args.n_steps),
        git_commit=np.array(commit),
        git_dirty=np.array(dirty),
        code_sha256=np.array(code_sha256),
        blank_frame_indices=np.array(blank_used, dtype=np.int32),
    )
    n_empty = int((det_count == 0).sum())
    n_distinct_ids = len({int(i) for i in trk_id.ravel() if i >= 0})
    print(f"wrote {out}  frames={len(det_count)}  total_dets={int(det_count.sum())}  "
          f"total_tracks={int(trk_count.sum())}  distinct_ids={n_distinct_ids}  "
          f"empty_det_frames={n_empty}")
    print(f"provenance: seed={args.seed}  n_steps={args.n_steps}  "
          f"git_commit={commit}  git_dirty={dirty}")
    print(f"code_sha256={code_sha256}")
    print(f"gap survival: track id {result['survivor_id']} confirmed both before and after "
          f"frames {blank_used}")
    print(f"min assignment-uniqueness margin: {result['min_assignment_margin']!r}")
    print(f"min |iou - iou_gate|: {result['min_gate_margin']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
