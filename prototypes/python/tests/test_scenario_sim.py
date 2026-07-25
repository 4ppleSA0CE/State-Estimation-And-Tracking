"""Sim is deterministic per seed, tracks 4 targets, and contains the engineered A/B crossing."""
import numpy as np
from scenario_sim import SimConfig, simulate


def test_deterministic_and_targets():
    cfg = SimConfig()
    f1, gt1 = simulate(cfg, seed=0)
    f2, gt2 = simulate(cfg, seed=0)
    assert len(f1) == cfg.n_steps
    assert all(np.array_equal(a, b) for a, b in zip(f1, f2))     # deterministic
    assert set(gt1[0].keys()) == {0, 1, 2, 3}                    # 4 targets


def test_crossing_exists():
    cfg = SimConfig()
    _, gt = simulate(cfg, seed=0)
    # targets 0 (A) and 1 (B) are engineered to cross near x=50
    dists = [np.hypot(gt[k][0][0] - gt[k][1][0], gt[k][0][1] - gt[k][1][1]) for k in range(cfg.n_steps)]
    assert min(dists) < 2.0                                      # they pass within 2 m
