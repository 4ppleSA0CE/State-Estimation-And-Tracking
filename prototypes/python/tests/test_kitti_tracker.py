# prototypes/python/tests/test_kitti_tracker.py
"""Two separated moving Car boxes -> two confirmed tracks, stable ids, via 3D-IoU association."""
from kitti_boxes import Box3D
from kitti_tracker import KittiTracker, KittiTrackerConfig


def _car(x, z, tid=-1):
    return Box3D(x=x, y=1.2, z=z, yaw=0.0, l=4.0, w=2.0, h=1.5, score=1.0, track_id=tid)


def test_two_targets_confirm_stable_ids():
    trk = KittiTracker(KittiTrackerConfig(min_hits=3, max_age=2))
    confirmed = []
    for k in range(8):
        dets = [_car(1.0 * k, 0.0), _car(1.0 * k, 20.0)]     # two cars, 20 m apart, moving +x
        confirmed = trk.step(dets)
    assert len(confirmed) == 2
    assert len({t.id for t in confirmed}) == 2


def test_empty_frame_safe():
    trk = KittiTracker()
    assert trk.step([]) == []


def test_maha_cost_flag_runs():
    trk = KittiTracker(KittiTrackerConfig(cost="maha", min_hits=1, max_age=2))
    out = None
    for k in range(3):
        out = trk.step([_car(1.0 * k, 0.0)])
    assert len(out) == 1
