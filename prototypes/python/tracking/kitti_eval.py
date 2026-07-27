"""py-motmetrics over KITTI Car tracks. Distance = 1 - iou_3d, gated to NaN below iou_thresh
(NaN = 'cannot match'). Hypotheses are per-frame (id, Box3D) snapshots (Box3D from track.box()
is a fresh object, so no mutable-state aliasing).

PIN before trusting any headline MOTA vs AB3DMOT: (1) `iou_thresh` (default 0.25) must match
KITTI/AB3DMOT's eval convention (0.25 / 0.5 / 0.7); (2) NO detection score-threshold filtering is
applied yet — the design's fixed-score-threshold MOTA is not implemented here (real AB3DMOT
detections carry scores; GT-as-detections are score=1). Set `KittiTrackingConfig.min_score` and
pin the threshold before reporting a comparable number."""
from __future__ import annotations

import motmetrics as mm
import numpy as np

from kitti_boxes import Box3D, iou_3d


def _dist(gt: list[Box3D], hyp: list[Box3D], iou_thresh: float) -> np.ndarray:
    if not gt or not hyp:
        return np.empty((len(gt), len(hyp)))
    d = np.full((len(gt), len(hyp)), np.nan)
    for i, g in enumerate(gt):
        for j, h in enumerate(hyp):
            iou = iou_3d(g, h)
            if iou >= iou_thresh:
                d[i, j] = 1.0 - iou
    return d


def accumulate(gt_frames, hyp_frames, iou_thresh: float = 0.25) -> "mm.MOTAccumulator":
    """gt_frames[k]: list[Box3D] with .track_id; hyp_frames[k]: list of (id, Box3D) snapshots."""
    acc = mm.MOTAccumulator(auto_id=True)
    for gt, hyp in zip(gt_frames, hyp_frames):
        gt_ids = [g.track_id for g in gt]
        h_ids = [hid for hid, _ in hyp]
        h_boxes = [b for _, b in hyp]
        acc.update(gt_ids, h_ids, _dist(gt, h_boxes, iou_thresh))
    return acc


def evaluate(gt_frames, hyp_frames, iou_thresh: float = 0.25):
    """Single-sequence metrics Series (mota, motp, idf1, num_switches)."""
    acc = accumulate(gt_frames, hyp_frames, iou_thresh)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=["mota", "motp", "idf1", "num_switches"], name="kitti").iloc[0]
