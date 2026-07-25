"""Gated Mahalanobis association: correct split, gate rejects far detections, greedy agrees."""
import numpy as np
from association import associate, DEFAULT_GATE


def _preds():
    # two tracks predicting (0,0) and (10,0), unit innovation covariance
    s = np.eye(2)
    return [(np.array([0.0, 0.0]), s), (np.array([10.0, 0.0]), s)]


def test_clean_two_by_two():
    dets = np.array([[10.1, 0.0], [0.1, 0.0]])          # near track 1, then track 0
    matches, un_d, un_t = associate(_preds(), dets, gate=DEFAULT_GATE)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert un_d == [] and un_t == []


def test_gate_rejects_far_detection():
    dets = np.array([[0.1, 0.0], [500.0, 500.0]])       # second is far outside every gate
    matches, un_d, un_t = associate(_preds(), dets, gate=DEFAULT_GATE)
    assert (0, 0) in matches
    assert 1 in un_d                                    # clutter stays unmatched
    assert 1 in un_t                                    # track 1 got nothing


def test_greedy_matches_hungarian_here():
    dets = np.array([[0.1, 0.0], [10.1, 0.0]])
    m_h, _, _ = associate(_preds(), dets, greedy=False)
    m_g, _, _ = associate(_preds(), dets, greedy=True)
    assert sorted(m_h) == sorted(m_g)
