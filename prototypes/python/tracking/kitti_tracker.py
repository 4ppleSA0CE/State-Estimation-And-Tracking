# prototypes/python/tracking/kitti_tracker.py
"""KITTI Car tracker: BEV-center IMM (reused unchanged) + carried box geometry; 3D-IoU
association (Mahalanobis behind a flag); M-of-N birth / max-age death. Emits Box3D tracks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from association import BIG_COST, associate_from_cost, mahalanobis_sq
from imm_filter import IMMConfig, IMMFilter
from kitti_boxes import Box3D, iou_3d

TENTATIVE, CONFIRMED, DEAD = "tentative", "confirmed", "dead"
_MAHA_GATE = 9.21     # chi2.ppf(0.99, 2)


def _default_imm() -> IMMConfig:
    # 10 Hz KITTI; sigma_pos ~ detection center noise (m); q_accel for road vehicles; CT bank for turns.
    return IMMConfig(dt=0.1, sigma_pos=0.5, q_accel=2.0, omegas=(0.2, -0.2))


@dataclass
class KittiTrackerConfig:
    imm: IMMConfig = field(default_factory=_default_imm)
    cost: str = "iou"           # "iou" | "maha"
    iou_gate: float = 0.01      # min IoU to allow a match (PIN to AB3DMOT's setting for the headline run)
    min_hits: int = 3
    max_age: int = 2
    greedy: bool = False
    p0_vel: float = 10.0        # birth velocity-uncertainty for KITTI init; BoxTrack uses THIS, not imm.p0_vel


class BoxTrack:
    def __init__(self, track_id: int, det: Box3D, cfg: KittiTrackerConfig) -> None:
        self.id = track_id
        self.cfg = cfg
        self.imm = IMMFilter(cfg.imm, np.eye(2) * cfg.imm.sigma_pos**2)
        self.imm.init_state(
            np.array([det.x, det.z, 0.0, 0.0]),
            np.diag([cfg.imm.sigma_pos**2, cfg.imm.sigma_pos**2, cfg.p0_vel, cfg.p0_vel]),
        )
        self.y, self.l, self.w, self.h, self.yaw = det.y, det.l, det.w, det.h, det.yaw
        self.score = det.score
        self.status = TENTATIVE
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.age = 0

    def _box(self, cx: float, cz: float) -> Box3D:
        return Box3D(cx, self.y, cz, self.yaw, self.l, self.w, self.h, self.score, self.id)

    def predict(self) -> None:
        self.imm.predict()
        self.age += 1

    def predicted_box(self) -> Box3D:
        x, _ = self.imm.state()
        return self._box(x[0], x[1])

    def box(self) -> Box3D:
        x, _ = self.imm.state()
        return self._box(x[0], x[1])

    def update(self, det: Box3D) -> None:
        self.imm.update(np.array([det.x, det.z]))
        self.y, self.l, self.w, self.h, self.yaw = det.y, det.l, det.w, det.h, det.yaw  # carry latest geometry
        self.score = det.score
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.imm.coast()
        self.hit_streak = 0
        self.time_since_update += 1


class KittiTracker:
    def __init__(self, cfg: KittiTrackerConfig | None = None) -> None:
        self.cfg = cfg or KittiTrackerConfig()
        self.tracks: list[BoxTrack] = []
        self._next_id = 0

    def _cost(self, dets: list[Box3D]) -> np.ndarray:
        n, m = len(self.tracks), len(dets)
        cost = np.full((n, m), BIG_COST)
        if self.cfg.cost == "iou":
            preds = [t.predicted_box() for t in self.tracks]
            for i, pb in enumerate(preds):
                for j, db in enumerate(dets):
                    iou = iou_3d(pb, db)
                    if iou >= self.cfg.iou_gate:
                        cost[i, j] = 1.0 - iou
        else:  # Mahalanobis on BEV centers
            for i, t in enumerate(self.tracks):
                z_pred, s = t.imm.predicted_measurement()
                for j, db in enumerate(dets):
                    d2 = mahalanobis_sq(np.array([db.x, db.z]) - z_pred, s)
                    if d2 <= _MAHA_GATE:
                        cost[i, j] = d2
        return cost

    def step(self, detections: list[Box3D]) -> list[BoxTrack]:
        for t in self.tracks:
            t.predict()
        if self.tracks and detections:
            cost = self._cost(detections)
            matches, un_d, un_t = associate_from_cost(cost, BIG_COST, self.cfg.greedy)
        else:
            matches, un_d, un_t = [], list(range(len(detections))), list(range(len(self.tracks)))
        for i, j in matches:
            self.tracks[i].update(detections[j])
        for i in un_t:
            self.tracks[i].mark_missed()
        for j in un_d:
            self.tracks.append(BoxTrack(self._next_id, detections[j], self.cfg))
            self._next_id += 1
        for t in self.tracks:
            if t.status == TENTATIVE and t.hits >= self.cfg.min_hits:
                t.status = CONFIRMED
            if t.time_since_update > self.cfg.max_age:
                t.status = DEAD
        self.tracks = [t for t in self.tracks if t.status != DEAD]
        return [t for t in self.tracks if t.status == CONFIRMED]
