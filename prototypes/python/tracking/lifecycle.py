"""TrackManager: M-of-N birth, max-age death, confirmed-only output. Owns track ids."""
from __future__ import annotations

import numpy as np
from imm_filter import IMMConfig, IMMFilter
from track import CONFIRMED, DEAD, TENTATIVE, Track


class TrackManager:
    def __init__(self, imm_cfg: IMMConfig, r, min_hits=3, max_age=2) -> None:
        self.imm_cfg = imm_cfg
        self.r = np.asarray(r, dtype=float)
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks: list[Track] = []
        self._next_id = 0

    def _new_imm(self, det: np.ndarray) -> IMMFilter:
        imm = IMMFilter(self.imm_cfg, self.r)
        x0 = np.array([det[0], det[1], 0.0, 0.0])                       # position known, velocity unknown
        p0 = np.diag([self.imm_cfg.sigma_pos**2, self.imm_cfg.sigma_pos**2,
                      self.imm_cfg.p0_vel, self.imm_cfg.p0_vel])
        imm.init_state(x0, p0)
        return imm

    def birth(self, unmatched_dets, dets) -> None:
        dets = np.asarray(dets, dtype=float).reshape(-1, 2)
        for j in unmatched_dets:
            self.tracks.append(Track(self._next_id, self._new_imm(dets[j])))
            self._next_id += 1

    def confirm_and_prune(self) -> None:
        for t in self.tracks:
            if t.status == TENTATIVE and t.hits >= self.min_hits:
                t.status = CONFIRMED
            if t.time_since_update > self.max_age:
                t.status = DEAD
        self.tracks = [t for t in self.tracks if t.status != DEAD]

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.status == CONFIRMED]
