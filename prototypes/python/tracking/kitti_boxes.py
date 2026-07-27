# prototypes/python/tracking/kitti_boxes.py
"""3D box (KITTI camera coords) + oriented 3D-IoU for tracking. BEV = ground plane (x, z);
y is down, box bottom at y and top at y-h. IoU = BEV-polygon overlap x height overlap / union."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Box3D:
    x: float
    y: float
    z: float
    yaw: float          # KITTI rotation_y (rad)
    l: float
    w: float
    h: float
    score: float = 1.0
    track_id: int = -1

    @property
    def center_bev(self) -> tuple[float, float]:
        return (self.x, self.z)


def _bev_corners(b: Box3D):
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    hl, hw = b.l / 2.0, b.w / 2.0
    local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return [(c * xl + s * zl + b.x, -s * xl + c * zl + b.z) for xl, zl in local]


def _ensure_ccw(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        s += x1 * z2 - x2 * z1
    return poly if s >= 0.0 else poly[::-1]


def _poly_area(poly):
    n = len(poly)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        a += x1 * z2 - x2 * z1
    return abs(a) * 0.5


def _clip(subject, clip):
    """Sutherland-Hodgman: clip `subject` by convex, CCW `clip`. Returns clipped polygon."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

    def isect(p1, p2, a, b):
        x1, z1 = p1
        x2, z2 = p2
        x3, z3 = a
        x4, z4 = b
        den = (x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1 - x3) * (z3 - z4) - (z1 - z3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), z1 + t * (z2 - z1))

    output = list(subject)
    n = len(clip)
    for i in range(n):
        a = clip[i]
        b = clip[(i + 1) % n]
        inp = output
        output = []
        if not inp:
            break
        for j in range(len(inp)):
            cur = inp[j]
            prv = inp[j - 1]
            if inside(cur, a, b):
                if not inside(prv, a, b):
                    output.append(isect(prv, cur, a, b))
                output.append(cur)
            elif inside(prv, a, b):
                output.append(isect(prv, cur, a, b))
    return output


def _bev_intersection_area(a: Box3D, b: Box3D) -> float:
    return _poly_area(_clip(_ensure_ccw(_bev_corners(a)), _ensure_ccw(_bev_corners(b))))


def iou_bev(a: Box3D, b: Box3D) -> float:
    inter = _bev_intersection_area(a, b)
    union = a.l * a.w + b.l * b.w - inter
    return inter / union if union > 0.0 else 0.0


def iou_3d(a: Box3D, b: Box3D) -> float:
    inter_area = _bev_intersection_area(a, b)
    lo = max(a.y - a.h, b.y - b.h)
    hi = min(a.y, b.y)
    h_overlap = max(0.0, hi - lo)
    inter_vol = inter_area * h_overlap
    union = a.l * a.w * a.h + b.l * b.w * b.h - inter_vol
    return inter_vol / union if union > 0.0 else 0.0
