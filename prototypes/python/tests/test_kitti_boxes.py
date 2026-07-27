# prototypes/python/tests/test_kitti_boxes.py
"""Oriented 3D-IoU on hand-computed cases (camera coords: BEV=(x,z), y down, box bottom at y)."""
import math
import numpy as np
from kitti_boxes import Box3D, iou_3d, iou_bev


def _box(x=0.0, z=0.0, yaw=0.0, l=4.0, w=2.0, h=1.5, y=0.0):
    return Box3D(x=x, y=y, z=z, yaw=yaw, l=l, w=w, h=h)


def test_identical_box_iou_one():
    a = _box()
    assert iou_bev(a, _box()) == 1.0
    assert abs(iou_3d(a, _box()) - 1.0) < 1e-9


def test_disjoint_box_iou_zero():
    assert iou_3d(_box(), _box(x=100.0, z=100.0)) == 0.0


def test_half_bev_overlap():
    a, b = _box(), _box(x=2.0)
    assert abs(iou_bev(a, b) - 1.0 / 3.0) < 1e-9
    assert abs(iou_3d(a, b) - 1.0 / 3.0) < 1e-9


def test_rotation_invariance_same_box():
    a = _box(yaw=0.5)
    assert abs(iou_3d(a, _box(yaw=0.5)) - 1.0) < 1e-9


def test_height_offset_reduces_iou():
    a, b = _box(), _box(y=0.75)
    assert abs(iou_3d(a, b) - 1.0 / 3.0) < 1e-9


def test_45_degree_partial_between_zero_and_one():
    a, b = _box(), _box(yaw=math.pi / 4)
    v = iou_bev(a, b)
    assert 0.0 < v < 1.0
