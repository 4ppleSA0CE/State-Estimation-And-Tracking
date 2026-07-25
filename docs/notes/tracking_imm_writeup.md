# IMM Multi-Target Tracker (synthetic)

The project's "where are they" half: an IMM filter per track, GNN data association, and an
M-of-N / max-age lifecycle, proven on a synthetic multi-target scenario before any KITTI or
C++ work.

## What it does
- Per-track IMM (CV + CA + CT±ω bank) lifted from `imm_synthetic.py`; a `legacy` config
  reproduces that prototype exactly (parity test to 1e-9), a `tracker` config adds the ω-bank
  and a missed-detection coast path.
- χ²-gated Mahalanobis association (gate ≈ 9.21, χ² 0.99 at 2 DOF) solved with the Hungarian
  algorithm (AB3DMOT-style greedy fallback available).
- M-of-N birth (M=3) rejects clutter; max-age death (3 misses) prunes lost tracks;
  confirmed-only output.

## Scenario
4 targets over 200 frames at 10 Hz: two constant-velocity targets (A, B) engineered to cross
at x≈50 m, one coordinated-turn target (ω=0.25 rad/s), one constant-acceleration target.
Detection probability p_detect=0.9 (missed detections), Poisson clutter λ=2 points/frame.
Detections are unlabeled 2D positions, shuffled per frame.

## Results (py-motmetrics)
MOTA=0.944, MOTP=0.543 m, IDF1=0.834, ID-switches=3. Knobs: gate=9.21, min_hits=3, max_age=3,
ω-bank=±0.25 rad/s, clutter λ=2. The few switches are dominated by the engineered A/B crossing,
where two position-only detections at identical y are genuinely ambiguous to a GNN gate.

![](../images/tracking_imm_summary.png)

## Why a fixed-ω CT bank, not estimated CTRV
Model competition handles the unknown turn rate while keeping the parity oracle against
`imm_synthetic.py` clean and the per-mode filters unchanged. Estimated CTRV (turn rate as a
state) is a noted future ablation; the ω-grid gap between bank members is its motivation.

## Note for the C++ port
Evaluation must snapshot each track's `(id, position)` at the frame it is produced — `Track`/IMM
state mutates in place, so holding references and reading positions after the run yields every
track's final state (a bug caught and fixed during this build). Stage 5's `kf_tracker` evaluator
faces the same trap.

## Next
KITTI Tracking (real 3D boxes, 3D-IoU association, an AB3DMOT comparison) — Stage 5 prep — then
the C++ ROS2 port (`kf_tracker`).
