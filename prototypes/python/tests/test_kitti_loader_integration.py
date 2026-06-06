from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from kitti_loader import KittiLoaderConfig, load_kitti_sequence


KITTI_ROOT = Path("data/kitti_raw")
KITTI_DATE = "2011_09_26"
KITTI_DRIVE = "0001"


def _has_kitti_sequence() -> bool:
    drive_path = KITTI_ROOT / KITTI_DATE / f"{KITTI_DATE}_drive_{KITTI_DRIVE}_sync"
    oxts_data = drive_path / "oxts" / "data"
    return (
        importlib.util.find_spec("pykitti") is not None
        and (drive_path / "oxts" / "timestamps.txt").exists()
        and oxts_data.is_dir()
        and any(oxts_data.glob("*.txt"))
    )


@pytest.mark.skipif(
    not _has_kitti_sequence(),
    reason="local KITTI Raw sequence is not installed",
)
def test_load_kitti_sequence_with_local_kitti_raw_data(tmp_path: Path) -> None:
    config = KittiLoaderConfig(
        root=KITTI_ROOT,
        date=KITTI_DATE,
        drive=KITTI_DRIVE,
        cache_root=tmp_path / "cache",
        force_refresh=True,
    )

    sequence = load_kitti_sequence(config)
    sample_count = sequence.sample_count

    assert sample_count > 0
    assert sequence.timestamps.shape == (sample_count,)
    assert sequence.lat_lon_alt.shape == (sample_count, 3)
    assert sequence.enu_position_m.shape == (sample_count, 3)
    assert sequence.roll_pitch_yaw.shape == (sample_count, 3)
    assert sequence.velocity.shape == (sample_count, 3)
    assert sequence.accel_body.shape == (sample_count, 3)
    assert sequence.gyro_body.shape == (sample_count, 3)
    assert sequence.gps_covariance.shape == (sample_count, 3, 3)
    assert np.isfinite(sequence.enu_position_m).all()
    np.testing.assert_allclose(sequence.enu_position_m[0], np.zeros(3), atol=1e-6)
