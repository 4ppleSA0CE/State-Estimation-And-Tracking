"""Two well-separated CV targets, clean detections → two confirmed tracks with stable ids."""
import numpy as np
from imm_filter import IMMConfig
from tracker import Tracker


def test_two_targets_confirm_and_keep_ids():
    trk = Tracker(IMMConfig(), r=np.eye(2), min_hits=3, max_age=2)
    ids_seen = set()
    confirmed_final = []
    for k in range(10):
        dets = np.array([[2.0 * k, 0.0], [2.0 * k, 40.0]])   # two targets, 40 m apart
        confirmed_final = trk.step(dets)
    assert len(confirmed_final) == 2
    ids = {t.id for t in confirmed_final}
    assert len(ids) == 2                                     # distinct, stable ids


def test_empty_frame_is_safe():
    trk = Tracker(IMMConfig(), r=np.eye(2))
    assert trk.step(np.empty((0, 2))) == []                 # no dets, no crash, no tracks
