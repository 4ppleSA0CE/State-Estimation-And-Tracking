# prototypes/python/tests/test_kitti_demo.py
"""Runs the demo on real data if present; otherwise skips (mirrors KITTI-raw integration tests)."""
import pytest
from kitti_tracking_loader import KittiTrackingConfig


def test_demo_runs_if_data_present():
    cfg = KittiTrackingConfig()
    if not cfg.label_dir.exists():
        pytest.skip(f"KITTI Tracking data not present at {cfg.label_dir}; download to run 5A end-to-end.")
    from kitti_demo import run_val_split
    summary = run_val_split(cfg, make_figure=False)
    assert "mota" in summary
