# prototypes/python/tracking/eval.py
"""py-motmetrics wrapper: accumulate GT vs tracker output per frame → MOTA/MOTP/IDF1/switches.
Distances are Euclidean, gated to NaN beyond max_d (NaN = 'cannot match' to motmetrics).
Hypotheses are per-frame (track_id, (x, y)) snapshots captured at step time — NOT live Track
objects, whose in-place mutation would otherwise read back as their final state."""
from __future__ import annotations

import motmetrics as mm
import numpy as np


def _dist_matrix(gt_pts: np.ndarray, hyp_pts: np.ndarray, max_d: float) -> np.ndarray:
    if len(gt_pts) == 0 or len(hyp_pts) == 0:
        return np.empty((len(gt_pts), len(hyp_pts)))
    d = np.linalg.norm(gt_pts[:, None, :] - hyp_pts[None, :, :], axis=2)
    d[d > max_d] = np.nan
    return d


def evaluate(gt_frames, hyp_frames, max_d: float = 4.0):
    """gt_frames[k]: {gt_id: (x,y)}; hyp_frames[k]: list of (track_id, (x,y)) captured at
    frame k. Returns a pandas Series with mota, motp, idf1, num_switches."""
    acc = mm.MOTAccumulator(auto_id=True)
    for gt, hyps in zip(gt_frames, hyp_frames):
        gt_ids = list(gt.keys())
        gt_pts = np.array([gt[i] for i in gt_ids], dtype=float) if gt_ids else np.empty((0, 2))
        h_ids = [hid for hid, _ in hyps]
        h_pts = np.array([pos for _, pos in hyps], dtype=float) if hyps else np.empty((0, 2))
        acc.update(gt_ids, h_ids, _dist_matrix(gt_pts, h_pts, max_d))
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "motp", "idf1", "num_switches"], name="sim")
    return summary.iloc[0]
