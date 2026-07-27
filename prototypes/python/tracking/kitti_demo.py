# prototypes/python/tracking/kitti_demo.py
"""Run the IMM Car tracker over the AB3DMOT val split, print aggregate MOTA/MOTP/IDF1/ID-sw,
and save a BEV figure. Reports the AB3DMOT real-detection headline and the GT-detection ceiling."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))          # tracking/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # prototypes/python/

import matplotlib.pyplot as plt
import motmetrics as mm

from kitti_eval import accumulate
from kitti_tracker import KittiTracker, KittiTrackerConfig
from kitti_tracking_loader import KittiTrackingConfig, load_detections, load_gt, val_sequences

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def _run_sequence(seq: str, data_cfg: KittiTrackingConfig, trk_cfg: KittiTrackerConfig):
    gt = load_gt(seq, data_cfg)
    dets = load_detections(seq, data_cfg)
    trk = KittiTracker(trk_cfg)
    n_frames = max(max(gt) if gt else 0, max(dets) if dets else 0) + 1
    gt_frames, hyp_frames = [], []
    for f in range(n_frames):
        frame_dets = dets.get(f, [])
        confirmed = trk.step(frame_dets)
        hyp_frames.append([(t.id, t.box()) for t in confirmed])   # snapshot at step time
        gt_frames.append(gt.get(f, []))
    return gt_frames, hyp_frames


def run_val_split(data_cfg: KittiTrackingConfig | None = None, trk_cfg: KittiTrackerConfig | None = None,
                  make_figure: bool = True) -> dict:
    data_cfg = data_cfg or KittiTrackingConfig()
    trk_cfg = trk_cfg or KittiTrackerConfig()
    accs, names, first = [], [], None
    for seq in val_sequences():
        gt_frames, hyp_frames = _run_sequence(seq, data_cfg, trk_cfg)
        accs.append(accumulate(gt_frames, hyp_frames))
        names.append(seq)
        if first is None:
            first = (gt_frames, hyp_frames)
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=["mota", "motp", "idf1", "num_switches"],
                              generate_overall=True)
    overall = summary.loc["OVERALL"]
    print(summary.to_string())
    print(f"OVERALL MOTA={overall['mota']:.3f} MOTP={overall['motp']:.3f} "
          f"IDF1={overall['idf1']:.3f} switches={int(overall['num_switches'])}")
    if make_figure and first is not None:
        _figure(*first)
    return {"mota": float(overall["mota"]), "motp": float(overall["motp"]),
            "idf1": float(overall["idf1"]), "num_switches": int(overall["num_switches"])}


def _figure(gt_frames, hyp_frames) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    for gt in gt_frames:
        for b in gt:
            ax.scatter(b.x, b.z, s=4, c="k", alpha=0.3)
    for hyp in hyp_frames:
        for tid, b in hyp:
            ax.scatter(b.x, b.z, s=6, c=f"C{tid % 10}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.set_aspect("equal", adjustable="box")
    ax.set_title("KITTI Car IMM tracker (BEV) — GT (black) vs tracks (color)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kitti_tracking_summary.png", dpi=150)
    print(f"saved {OUTPUT_DIR / 'kitti_tracking_summary.png'}")


if __name__ == "__main__":
    run_val_split()
