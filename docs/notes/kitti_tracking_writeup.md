# IMM Tracker on KITTI Tracking (Car)

The synthetic-scenario tracker, run on real KITTI Tracking Car data — the tracking analog of
the "ESKF on KITTI" step, and the last checkpoint before the C++ ROS2 port.

## Setup
- BEV-center IMM (CV + CA + CT±ω) filtering ground-plane motion; box size/yaw carried from the
  matched detection; reconstructed 3D boxes out. The IMM (`imm_filter.py`) is reused unchanged.
- 3D-IoU association (Sutherland-Hodgman BEV overlap × height), Hungarian; Mahalanobis ablation.
- Camera-frame, per-frame (AB3DMOT parity; world-frame ego-coupling comes later, in the coupled
  pipeline).
- Car only, AB3DMOT val split. Detections: AB3DMOT's Car files (headline) + GT-as-detections (ceiling).
- Driver: `scripts/run_tracking_eval.py`; numbers in `data/results/tracking_per_sequence.csv` and
  `data/results/tracking_summary.csv`. Whole grid runs in ~30 s.

## Results (py-motmetrics, 3D-IoU threshold 0.25, 11 val sequences, 9550 Car GT boxes)
| detections | MOTA | mean 3D IoU | IDF1 | ID-sw | FP | FN | recall |
|---|---|---|---|---|---|---|---|
| AB3DMOT real (PointRCNN, ungated) | 0.042 | 0.724 | 0.619 | 31 | 8039 | 1075 | 0.887 |
| GT (ceiling) | 0.886 | 0.854 | 0.936 | 5 | 347 | 739 | 0.923 |

**These are fixed-threshold MOTA numbers and are not comparable to AB3DMOT's published figure.**
AB3DMOT's headline integrates over detection-score thresholds (sAMOTA), and the MOTA in their table
is taken at their chosen operating point; the run above feeds *every* PointRCNN detection to the
tracker and reports MOTA at that single, deliberately ungated operating point. The gap below is
therefore not a quality comparison — see "Why the headline MOTA is 0.04".

For reference, AB3DMOT's own Car val numbers at the same 0.25 3D-IoU threshold, from
`docs/KITTI.md` in their repo (`xinshuoweng/AB3DMOT`, master): sAMOTA 93.34, MOTA 86.47,
MOTP 79.40, IDS 0, FRAG 15, FP 368, FN 766.

The GT row is a **ceiling, not a result**: it is what the filter and the birth/death logic can do
when detection error is removed entirely. It does not reach 1.0 because `min_hits=3` withholds each
track for its first two frames (FN) and `max_age=2` coasts each track two frames past its last
detection (FP) — 739 misses and 347 false positives that are pure track-lifecycle cost, plus IMM
lag that pulls mean IoU to 0.854 instead of 1.0.

### Per sequence
| seq | AB3DMOT MOTA | AB3DMOT IDF1 | AB3DMOT ID-sw | GT MOTA | GT IDF1 | GT ID-sw | Car GT |
|---|---|---|---|---|---|---|---|
| 0001 | 0.325 | 0.686 | 13 | 0.843 | 0.905 | 3 | 2681 |
| 0006 | 0.493 | 0.700 | 2 | 0.902 | 0.944 | 1 | 550 |
| 0008 | 0.242 | 0.640 | 3 | 0.831 | 0.909 | 0 | 1046 |
| 0010 | 0.237 | 0.670 | 1 | 0.756 | 0.865 | 0 | 603 |
| 0012 | 0.188 | 0.626 | 1 | 0.958 | 0.979 | 0 | 144 |
| 0013 | **-11.018** | 0.093 | 1 | 0.891 | 0.944 | 0 | 55 |
| 0014 | 0.422 | 0.717 | 1 | 0.730 | 0.841 | 1 | 455 |
| 0015 | 0.280 | 0.707 | 2 | 0.958 | 0.979 | 0 | 899 |
| 0016 | 0.392 | 0.654 | 2 | 0.990 | 0.995 | 0 | 836 |
| 0018 | 0.470 | 0.763 | 5 | 0.938 | 0.969 | 0 | 1354 |
| 0019 | **-2.118** | 0.381 | 0 | 0.972 | 0.986 | 0 | 927 |
| **pooled** | **0.042** | **0.619** | **31** | **0.886** | **0.936** | **5** | **9550** |

Pooled, not averaged: MOTA is pooled error counts over pooled object counts, so the OVERALL row is
computed from all 11 accumulators at once (`compute_many(..., generate_overall=True)`). Averaging
the per-sequence MOTAs here would give -0.917, which is meaningless — it lets 0013's 55 Car boxes
outweigh 0001's 2681.

### Why the headline MOTA is 0.04
Recall is fine — 88.7%, and 1075 misses against AB3DMOT's 766. The score is destroyed entirely by
8039 false positives (22× their 368), from two causes, neither of which is a tracker defect:

1. **No detection-score gate** (pinned below). PointRCNN's scores here are unnormalised, spanning
   -0.85 to 15.69 across the split, and the low end is mostly noise. The median detection score is 0.38 in
   0013 and 0.93 in 0019 — the two sequences with negative MOTA.
2. **No KITTI don't-care handling.** Official KITTI eval ignores hypotheses landing in DontCare
   regions and treats Van as neutral for Car. The frozen eval drops every non-Car GT row at parse
   time, so a detection sitting on a Van or in a DontCare region is scored as a hard FP. This is
   exactly the shape of the two failures: 0013 has 55 Car GT rows against 935 DontCare, 929
   Pedestrian and 69 Van; 0019 has 927 Car against 2366 DontCare and 486 Van. Both go negative
   because MOTA's numerator is unbounded while its denominator is only the Car count.

Score-threshold sensitivity, same tracker, same eval, gate applied to the detections (diagnostic
only — the headline stays ungated):

| detection score gate | MOTA | IDF1 | ID-sw | FP | FN | recall |
|---|---|---|---|---|---|---|
| none (headline) | 0.042 | 0.619 | 31 | 8039 | 1075 | 0.887 |
| ≥ 2 | 0.608 | 0.752 | 44 | 2199 | 1501 | 0.843 |
| ≥ 4 | 0.660 | 0.753 | 40 | 1183 | 2028 | 0.788 |
| ≥ 6 | 0.600 | 0.670 | 52 | 703 | 3069 | 0.679 |

So 0.617 MOTA of the 0.822 gap to AB3DMOT's 86.47 is the ungated tail alone. The rest is the
don't-care convention and their threshold sweep. Closing it properly means implementing KITTI's
ignore rules, not tuning the tracker.

## Ablation: 3D-IoU vs Mahalanobis association
Sequences 0001, 0008 and 0018 — the three densest in Car ground truth (5081 of the 9550 GT boxes).
Both costs pooled over the *same* three sequences; the 11-sequence pooled number is not a valid
baseline for a 3-sequence run.

| association | MOTA | mean 3D IoU | IDF1 | ID-sw | FP | FN | recall |
|---|---|---|---|---|---|---|---|
| 3D IoU (gate 0.01) | 0.347 | 0.729 | 0.697 | 21 | 2666 | 633 | 0.875 |
| Mahalanobis (χ² gate 9.21, 2 DoF) | 0.331 | 0.733 | 0.676 | 21 | 2497 | 882 | 0.826 |

Near-identical, and the same 21 ID switches. Mahalanobis trades recall for precision — 169 fewer
false positives, 249 more misses — because a BEV-centre χ² gate on a 0.5 m position sigma is
tighter than a 0.01 IoU gate for a 4 m car, so a lagging track drops its detection rather than
grabbing it. IoU wins on IDF1 (0.697 vs 0.676) since box overlap disambiguates neighbouring
parked cars that BEV centres alone do not. At this detection quality the association cost is not
the bottleneck; the false-positive rate is.

## Notes / honesty
- MOTA at a fixed score threshold, not AMOTA (AB3DMOT's headline integrates over thresholds).
  The table caption says this; the numbers must not be read as beating or losing to their figure.
- The GT row is a ceiling with perfect detections, not a result.
- Eval conventions, now pinned:
  - **3D-IoU match threshold 0.25** — AB3DMOT's Car convention, and already the default in the
    frozen `evaluate()`. Their published Car val table reports at 0.25 / 0.5 / 0.7; 0.25 is the row
    quoted above.
  - **Detection score threshold: none.** Every PointRCNN detection is fed to the tracker. AB3DMOT
    thresholds at *output*, not tracker input, and GT-as-detections all carry score 1.0 — any
    input-side gate would make the two detection sources incomparable. The consequence is the
    8039 FPs above, and it is the pinned choice, not an oversight.
  - **DontCare / non-Car: dropped at parse time.** The frozen `parse_label_file` keeps only `Car`
    rows, so DontCare, Van, Pedestrian, Cyclist, Person, Tram and Misc never enter the GT. This is
    *not* KITTI's official convention, which treats DontCare regions and Van as ignore rather than
    as absent, and it is the second reason the FP count is not comparable to theirs.
- The val-split sequence list is **verified**, no longer a pin:
  `['0001','0006','0008','0010','0012','0013','0014','0015','0016','0018','0019']` at
  `AB3DMOT_libs/utils.py` line 20 of `xinshuoweng/AB3DMOT` master
  (`if split == 'val': seq_eval = [...]`), byte-identical to `VAL_SEQUENCES` in
  `kitti_tracking_loader.py`. Corroborated locally: `data/kitti_tracking/ab3dmot_car_val/` holds
  detection files for exactly those 11 sequences, while `label_02/` holds all 21. Earlier notes
  cited this as line 39; the list contents match, the line number has moved.
- The AB3DMOT detection-file column layout is **verified** against the data: comma-separated, 15
  columns, `frame, type, x1, y1, x2, y2, score, h, w, l, x, y, z, rot_y, alpha`, consistent with
  their `dets = seq_dets[:, 7:14]`. This is *not* the whitespace-separated 17-column `label_02`
  layout, so the frozen `load_detections()` returns zero detections for these files; the driver
  parses them itself (`parse_ab3dmot_detections`).
- MOTP direction: py-motmetrics reports MOTP as a mean *distance* (1 − IoU, lower better), while
  KITTI/AB3DMOT tables report a mean IoU percentage (higher better). Both are in the CSVs as
  `motp_dist` and `motp_iou`; the tables above use mean 3D IoU.

## Next
Port this tracker to a C++ ROS2 node (`kf_tracker`), then couple it to the ESKF in world frame.
