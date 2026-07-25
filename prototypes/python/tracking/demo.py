# prototypes/python/tracking/demo.py
"""Run sim → tracker → eval, print DOD metrics, save output/tracking_imm_summary.png."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))          # tracking/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # prototypes/python/

import matplotlib.pyplot as plt
import numpy as np
from imm_filter import IMMConfig
from scenario_sim import SimConfig, simulate
from tracker import Tracker
from eval import evaluate

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def main() -> None:
    sim_cfg = SimConfig()
    frames, gt_frames = simulate(sim_cfg, seed=0)
    trk = Tracker(IMMConfig(omegas=(0.25, -0.25)), r=np.eye(2) * sim_cfg.sigma_pos**2,
                  min_hits=3, max_age=3)
    hyp_frames = [[(t.id, tuple(t.position())) for t in trk.step(f)] for f in frames]
    metrics = evaluate(gt_frames, hyp_frames)
    print(f"MOTA={metrics['mota']:.3f} MOTP={metrics['motp']:.3f} "
          f"IDF1={metrics['idf1']:.3f} switches={int(metrics['num_switches'])}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    for tid in gt_frames[0]:
        xy = np.array([gt[tid] for gt in gt_frames])
        ax.plot(xy[:, 0], xy[:, 1], "k-", alpha=0.4, linewidth=1)
    for hyps in hyp_frames:
        for tid, pos in hyps:
            ax.scatter(pos[0], pos[1], s=4, c=f"C{tid % 10}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"IMM tracker — MOTA={metrics['mota']:.2f}, ID-sw={int(metrics['num_switches'])}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tracking_imm_summary.png", dpi=150)
    print(f"saved {OUTPUT_DIR / 'tracking_imm_summary.png'}")


if __name__ == "__main__":
    main()
