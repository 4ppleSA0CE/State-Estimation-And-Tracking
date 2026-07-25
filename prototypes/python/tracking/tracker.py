# prototypes/python/tracking/tracker.py
"""Per-frame multi-target tracker: predict → gate/associate → update → coast → birth →
prune → emit confirmed. Ties IMMFilter + association + lifecycle together."""
from __future__ import annotations

import numpy as np
from association import DEFAULT_GATE, associate
from imm_filter import IMMConfig
from lifecycle import TrackManager


class Tracker:
    def __init__(self, imm_cfg: IMMConfig, r, min_hits=3, max_age=2,
                 gate=DEFAULT_GATE, greedy=False) -> None:
        self.mgr = TrackManager(imm_cfg, r, min_hits=min_hits, max_age=max_age)
        self.gate = gate
        self.greedy = greedy

    def step(self, detections):
        dets = np.asarray(detections, dtype=float).reshape(-1, 2)
        tracks = self.mgr.tracks

        for t in tracks:                                   # 1. predict all
            t.predict()
        preds = [t.predicted_measurement() for t in tracks]

        matches, unmatched_d, unmatched_t = associate(     # 2-3. gate + associate
            preds, dets, gate=self.gate, greedy=self.greedy
        )

        for i, j in matches:                               # 4. update matched
            tracks[i].update(dets[j])
        for i in unmatched_t:                              # 5. coast unmatched tracks
            tracks[i].mark_missed()

        self.mgr.birth(unmatched_d, dets)                  # 6. birth from unmatched dets
        self.mgr.confirm_and_prune()                       # 7. confirm + prune dead
        return self.mgr.confirmed()                        # 8. emit confirmed
