# prototypes/python/tracking/scenario_sim.py
"""Multi-target synthetic scenario: 4 targets (one engineered crossing), missed detections,
and Poisson clutter. Emits per-frame unlabeled detections + ground-truth id→position."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imm_synthetic import ct_propagate_state


@dataclass
class SimConfig:
    dt: float = 0.1
    n_steps: int = 200
    sigma_pos: float = 1.0
    p_detect: float = 0.9
    clutter_rate: float = 2.0                      # Poisson mean clutter points / frame
    scene: tuple[float, float, float, float] = (-20.0, 120.0, -60.0, 60.0)


def _truth(cfg: SimConfig) -> dict[int, np.ndarray]:
    """Ground-truth (x, y) per target over all frames."""
    n, dt = cfg.n_steps, cfg.dt
    t = np.arange(n) * dt
    truth: dict[int, np.ndarray] = {}
    # Target 0 (A): CV +x along y=0
    truth[0] = np.stack([8.0 * t, np.zeros(n)], axis=1)
    # Target 1 (B): CV -x along y=0 — crosses A at x=50, t=6.25 s
    truth[1] = np.stack([100.0 - 8.0 * t, np.zeros(n)], axis=1)
    # Target 2 (C): coordinated turn, start (0, 40) heading +x, ω=0.25 rad/s
    c = np.zeros((n, 4))
    c[0] = [0.0, 40.0, 8.0, 0.0]
    for k in range(1, n):
        c[k] = ct_propagate_state(c[k - 1], dt, 0.25)
    truth[2] = c[:, :2]
    # Target 3 (D): constant acceleration, start (0, -40)
    ax, ay, vx, vy, y0 = 0.3, 0.1, 6.0, 0.0, -40.0
    truth[3] = np.stack([vx * t + 0.5 * ax * t**2, y0 + vy * t + 0.5 * ay * t**2], axis=1)
    return truth


def simulate(cfg: SimConfig, seed: int):
    """Return (frames, gt_frames): frames[k] is an (m,2) detection array (order shuffled);
    gt_frames[k] is {target_id: (x, y)}."""
    rng = np.random.default_rng(seed)
    truth = _truth(cfg)
    xmin, xmax, ymin, ymax = cfg.scene
    frames, gt_frames = [], []
    for k in range(cfg.n_steps):
        gt: dict[int, tuple[float, float]] = {}
        dets: list = []
        for tid, arr in truth.items():
            p = arr[k]
            gt[tid] = (float(p[0]), float(p[1]))
            if rng.random() < cfg.p_detect:                       # else: missed detection
                dets.append(p + rng.normal(0.0, cfg.sigma_pos, size=2))
        for _ in range(rng.poisson(cfg.clutter_rate)):            # false positives
            dets.append([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        dets = np.asarray(dets, dtype=float).reshape(-1, 2)
        rng.shuffle(dets)                                         # detector gives no identity/order
        frames.append(dets)
        gt_frames.append(gt)
    return frames, gt_frames
