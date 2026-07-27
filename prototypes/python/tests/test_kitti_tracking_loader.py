"""Loader parses label_02 into Car Box3D lists; DontCare filtered; values correct."""
from pathlib import Path
import pytest
from kitti_tracking_loader import parse_label_file

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kitti_label_sample.txt"


def test_parses_cars_only_with_values():
    frames = parse_label_file(FIXTURE)
    assert set(frames.keys()) == {0}
    boxes = frames[0]
    assert len(boxes) == 2                       # DontCare dropped
    b0 = next(b for b in boxes if b.track_id == 0)
    assert (b0.h, b0.w, b0.l) == (1.5, 1.6, 4.0)
    assert (b0.x, b0.y, b0.z) == (3.0, 1.2, 12.0)
    assert abs(b0.yaw - (-1.57)) < 1e-9
