# prototypes/python/tests/test_tracker_e2e.py
"""DOD gate: on the synthetic scenario, MOTA ≥ 0.7 and ID-switches stay low. Hypotheses are
captured as (id, position) at step time (Track objects mutate in place across frames)."""
import numpy as np
from imm_filter import IMMConfig
from scenario_sim import SimConfig, simulate
from tracker import Tracker
from eval import evaluate


def test_dod_mota_and_switches():
    sim_cfg = SimConfig()
    frames, gt_frames = simulate(sim_cfg, seed=0)
    trk = Tracker(IMMConfig(omegas=(0.25, -0.25)), r=np.eye(2) * sim_cfg.sigma_pos**2,
                  min_hits=3, max_age=3)
    hyp_frames = [[(t.id, tuple(t.position())) for t in trk.step(f)] for f in frames]
    metrics = evaluate(gt_frames, hyp_frames)
    assert metrics["mota"] >= 0.7
    assert metrics["num_switches"] <= 4       # engineered A/B crossing dominates the few switches
