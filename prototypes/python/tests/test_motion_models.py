"""Bank builder returns the right modes in the right order."""
import numpy as np
from motion_models import build_model_bank


def test_bank_order_and_size():
    r = np.eye(2)
    filters, names = build_model_bank(dt=0.1, sigma_pos=1.0, q_accel=0.05, r=r, omegas=(0.25, -0.25))
    assert names == ["CV", "CA", "CT+0.25", "CT-0.25"]
    assert len(filters) == 4


def test_bank_no_ct():
    r = np.eye(2)
    filters, names = build_model_bank(dt=0.1, sigma_pos=1.0, q_accel=0.05, r=r, omegas=())
    assert names == ["CV", "CA"]
    assert len(filters) == 2
