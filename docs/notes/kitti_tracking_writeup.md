# IMM Tracker on KITTI Tracking (Car)

The synthetic Stage 4 tracker, run on real KITTI Tracking Car data — the tracking analog of
the Stage 1 "ESKF on KITTI" step, before the C++ ROS2 port (Stage 5B).

## Setup
- BEV-center IMM (CV + CA + CT±ω) filtering ground-plane motion; box size/yaw carried from the
  matched detection; reconstructed 3D boxes out. The IMM (`imm_filter.py`) is reused unchanged.
- 3D-IoU association (Sutherland-Hodgman BEV overlap × height), Hungarian; Mahalanobis ablation.
- Camera-frame, per-frame (AB3DMOT parity; world-frame ego-coupling is Stage 6).
- Car only, AB3DMOT val split. Detections: AB3DMOT's Car files (headline) + GT-as-detections (ceiling).

## Results (py-motmetrics, IoU threshold [pin]) — PENDING DATA until the KITTI run
| detections | MOTA | MOTP | IDF1 | ID-sw |
|---|---|---|---|---|
| AB3DMOT real | .. | .. | .. | .. |
| GT (ceiling) | .. | .. | .. | .. |

AB3DMOT published Car MOTA (their table): [pin]. Gap discussion: [ours vs theirs; IMM vs plain-CV].

## Ablation: 3D-IoU vs Mahalanobis association — PENDING DATA
[MOTA/ID-sw for cost="iou" vs cost="maha" on the same detections.]

## Notes / honesty
- MOTA at a fixed score threshold, not AMOTA (AB3DMOT's headline integrates over thresholds).
- Eval convention to pin before the headline: IoU threshold [x], score threshold [y], DontCare handling [z].
- The KITTI val-split sequence list and the AB3DMOT detection-file column layout must be verified
  against AB3DMOT's repo before the headline number is trusted.

## Next
Stage 5B — port this tracker to a C++ ROS2 node (`kf_tracker`). Stage 6 — world-frame ego-coupling.
