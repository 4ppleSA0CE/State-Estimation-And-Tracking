"""Track: one tracked object — an IMMFilter plus SORT/AB3DMOT-style lifecycle counters."""
from __future__ import annotations

import numpy as np
from imm_filter import IMMFilter

TENTATIVE, CONFIRMED, DEAD = "tentative", "confirmed", "dead"


class Track:
    def __init__(self, track_id: int, imm: IMMFilter) -> None:
        self.id = track_id
        self.imm = imm
        self.status = TENTATIVE
        self.hits = 1                 # born from one detection
        self.hit_streak = 1
        self.time_since_update = 0
        self.age = 0

    def predict(self) -> None:
        self.imm.predict()
        self.age += 1

    def update(self, z: np.ndarray) -> None:
        self.imm.update(z)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.imm.coast()
        self.hit_streak = 0
        self.time_since_update += 1

    def position(self) -> np.ndarray:
        return self.imm.state()[0][:2]

    def predicted_measurement(self):
        return self.imm.predicted_measurement()
