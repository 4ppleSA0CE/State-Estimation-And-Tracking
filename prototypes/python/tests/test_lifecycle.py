"""Track counters + M-of-N birth / max-age death: clutter never confirms, 3 hits confirms,
misses coast then kill, ids are unique and stable."""
import numpy as np
from imm_filter import IMMConfig
from lifecycle import TrackManager


def _mgr():
    return TrackManager(IMMConfig(), r=np.eye(2), min_hits=3, max_age=2)


def test_clutter_never_confirms_then_dies():
    mgr = _mgr()
    mgr.birth([0], np.array([[5.0, 5.0]]))          # one lone detection (looks like clutter)
    assert mgr.confirmed() == []                    # tentative, not confirmed
    for _ in range(4):                              # nothing ever matches it
        for t in mgr.tracks:
            t.predict()
            t.mark_missed()
        mgr.confirm_and_prune()
    assert mgr.tracks == []                          # died by max_age


def test_three_hits_confirms():
    mgr = _mgr()
    mgr.birth([0], np.array([[0.0, 0.0]]))
    trk = mgr.tracks[0]
    for step in range(2):                            # two more hits → hits == 3
        trk.predict()
        trk.update(np.array([step + 1.0, 0.0]))
        mgr.confirm_and_prune()
    assert trk.hits == 3
    assert trk in mgr.confirmed()


def test_ids_unique():
    mgr = _mgr()
    mgr.birth([0, 1], np.array([[0.0, 0.0], [50.0, 0.0]]))
    assert {t.id for t in mgr.tracks} == {0, 1}
