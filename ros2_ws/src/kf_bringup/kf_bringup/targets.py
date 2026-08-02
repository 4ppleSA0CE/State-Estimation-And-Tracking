"""Synthetic targets on the real KITTI ego path.

Four vehicles are placed relative to the ground-truth ego pose at t0, then propagated in ENU
by their own dynamics. Detections are produced by projecting ENU truth into base_link with the
GROUND-TRUTH ego pose (what a real sensor sees) -- the pipeline then converts them back with
the ESTIMATED pose, which is where localization error enters the tracker.

Pure numpy, no ROS import, so this unit-tests on the host.

Box layout everywhere in this module: [x, y, z, yaw, l, w, h], with z the box BOTTOM-CENTER
(the KITTI label convention), so the ENU->map_bev height mapping is a plain negation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

X, Y, Z, YAW, L, W, H = range(7)
BOX_DIM = 7

CAR_DIMS = (3.9, 1.6, 1.5)      # KITTI Car median (l, w, h), m
EGO_HEIGHT_M = 1.7              # OXTS/IMU origin above the ground on the KITTI platform
N_TARGETS = 4

# Placement in the ego frame at t0: (forward m, left m, speed m/s or None to match the ego,
# heading offset rad). Offset 0 = same direction as the ego, pi = oncoming, +pi/2 = crossing
# right-to-left.
_LAYOUT = (
    (25.0,   0.0, None, 0.0),            # 0 leading vehicle
    (60.0,   3.5,  8.0, math.pi),        # 1 oncoming, near lane
    (80.0,   7.0, 10.0, math.pi),        # 2 oncoming, far lane (starts outside the range gate)
    (40.0, -12.0,  6.0, math.pi / 2),    # 3 crossing -- the maneuver subject
)


@dataclass(frozen=True)
class TargetConfig:
    p_detect: float = 0.9
    det_pos_std: float = 0.35
    det_yaw_std: float = 0.03
    clutter_lambda: float = 0.0
    max_range_m: float = 60.0
    fov_deg: float = 90.0
    maneuver_target: int = -1            # -1 = no maneuver
    maneuver_start_s: float = 5.0        # seconds after the FIRST frame, not absolute
    maneuver_omega: float = 0.4          # rad/s

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_detect <= 1.0:
            raise ValueError(f"p_detect must be in [0, 1], got {self.p_detect}")
        if self.det_pos_std < 0.0:
            raise ValueError(f"det_pos_std must be >= 0, got {self.det_pos_std}")
        if self.det_yaw_std < 0.0:
            raise ValueError(f"det_yaw_std must be >= 0, got {self.det_yaw_std}")
        if self.clutter_lambda < 0.0:
            raise ValueError(f"clutter_lambda must be >= 0, got {self.clutter_lambda}")
        if self.max_range_m <= 0.0:
            raise ValueError(f"max_range_m must be > 0, got {self.max_range_m}")
        if not 0.0 < self.fov_deg <= 360.0:
            raise ValueError(f"fov_deg must be in (0, 360], got {self.fov_deg}")
        if not -1 <= self.maneuver_target < N_TARGETS:
            raise ValueError(f"maneuver_target must be -1 or 0..{N_TARGETS - 1}, "
                             f"got {self.maneuver_target}")


def _wrap(a):
    """Wrap angle(s) to the half-open branch [-pi, pi) -- _wrap(pi) is -pi, not +pi."""
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def _propagate(s: np.ndarray, dt: float, omega: float) -> np.ndarray:
    """One step of [x, y, vx, vy]. omega == 0 is CV; otherwise the exact coordinated turn.

    Mirrors imm_synthetic.ct_propagate_state, deliberately re-implemented rather than imported:
    ros2_ws code must not depend on prototypes/python at container runtime. The pinned
    speed-preservation test is what keeps the two honest.
    """
    x, y, vx, vy = float(s[0]), float(s[1]), float(s[2]), float(s[3])
    if abs(omega) < 1e-9:
        return np.array([x + vx * dt, y + vy * dt, vx, vy])
    wt = omega * dt
    sw, cw = math.sin(wt), math.cos(wt)
    return np.array([
        x + (vx * sw - vy * (1.0 - cw)) / omega,
        y + (vx * (1.0 - cw) + vy * sw) / omega,
        vx * cw - vy * sw,
        vx * sw + vy * cw,
    ])


def target_truth(cfg: TargetConfig, frame_t, ego_pos0, ego_yaw0: float,
                 ego_speed0: float) -> np.ndarray:
    """Ground-truth ENU boxes, shape (n_frames, N_TARGETS, BOX_DIM)."""
    frame_t = np.asarray(frame_t, dtype=float)
    ego_pos0 = np.asarray(ego_pos0, dtype=float)
    if frame_t.ndim != 1 or frame_t.size == 0:
        raise ValueError("frame_t must be a non-empty 1-D array")

    ground_z = float(ego_pos0[2]) - EGO_HEIGHT_M
    c0, s0 = math.cos(ego_yaw0), math.sin(ego_yaw0)

    state = np.zeros((N_TARGETS, 4))
    for i, (fwd, left, spd, dpsi) in enumerate(_LAYOUT):
        state[i, 0] = ego_pos0[0] + c0 * fwd - s0 * left
        state[i, 1] = ego_pos0[1] + s0 * fwd + c0 * left
        v = ego_speed0 if spd is None else spd
        psi = ego_yaw0 + dpsi
        state[i, 2] = v * math.cos(psi)
        state[i, 3] = v * math.sin(psi)

    out = np.zeros((frame_t.size, N_TARGETS, BOX_DIM))
    onset = float(frame_t[0]) + cfg.maneuver_start_s
    for k in range(frame_t.size):
        if k > 0:
            dt = float(frame_t[k] - frame_t[k - 1])
            for i in range(N_TARGETS):
                turning = (i == cfg.maneuver_target) and (frame_t[k] >= onset)
                state[i] = _propagate(state[i], dt, cfg.maneuver_omega if turning else 0.0)
        out[k, :, X] = state[:, 0]
        out[k, :, Y] = state[:, 1]
        out[k, :, Z] = ground_z
        out[k, :, YAW] = np.arctan2(state[:, 3], state[:, 2])
        out[k, :, L], out[k, :, W], out[k, :, H] = CAR_DIMS
    return out


def enu_to_base_link(boxes_enu, ego_pos, ego_rot) -> np.ndarray:
    """ENU boxes -> base_link (x forward, y left, z up), using the FULL ego rotation.

    Position uses all of R so roll/pitch are handled exactly. Box yaw uses only the heading
    component -- a BEV box yaw has no roll/pitch meaning, and this is the same decomposition
    ego_transform.hpp performs in the opposite direction (spec section 4.3).
    """
    boxes_enu = np.asarray(boxes_enu, dtype=float)
    ego_pos = np.asarray(ego_pos, dtype=float)
    ego_rot = np.asarray(ego_rot, dtype=float)
    if ego_rot.shape != (3, 3):
        raise ValueError(f"ego_rot must be 3x3, got {ego_rot.shape}")

    ego_yaw = math.atan2(ego_rot[1, 0], ego_rot[0, 0])
    out = boxes_enu.copy()
    delta = boxes_enu[..., [X, Y, Z]] - ego_pos
    out[..., [X, Y, Z]] = delta @ ego_rot            # (R^T d)^T == d^T R
    out[..., YAW] = _wrap(boxes_enu[..., YAW] - ego_yaw)
    return out


def enu_to_map_bev(boxes_enu) -> np.ndarray:
    """ENU -> map_bev: (x, y, z, yaw) -> (x, -z, y, -yaw). Spec section 4.3."""
    boxes_enu = np.asarray(boxes_enu, dtype=float)
    out = boxes_enu.copy()
    out[..., X] = boxes_enu[..., X]
    out[..., Y] = -boxes_enu[..., Z]
    out[..., Z] = boxes_enu[..., Y]
    out[..., YAW] = -boxes_enu[..., YAW]
    return out


def map_bev_to_enu(boxes_bev) -> np.ndarray:
    """map_bev -> ENU, the exact inverse of enu_to_map_bev."""
    boxes_bev = np.asarray(boxes_bev, dtype=float)
    out = boxes_bev.copy()
    out[..., X] = boxes_bev[..., X]
    out[..., Y] = boxes_bev[..., Z]
    out[..., Z] = -boxes_bev[..., Y]
    out[..., YAW] = -boxes_bev[..., YAW]
    return out


def detect(boxes_b, rng, cfg: TargetConfig):
    """Detector model on base_link boxes. Returns (dets (m, BOX_DIM), src_ids (m,)).

    src_ids carries the true originating target index, or -1 for clutter. It is recorded for
    offline plotting only -- the live pipeline publishes object_id = -1 for everything.

    RNG call order is part of the contract (visible targets in index order: detect draw, then
    position noise, then yaw noise; then the clutter count; then per-clutter draws; then one
    permutation). Changing the order changes every recorded run.
    """
    boxes_b = np.asarray(boxes_b, dtype=float).reshape(-1, BOX_DIM)
    half_fov = math.radians(cfg.fov_deg) * 0.5

    dets: list[np.ndarray] = []
    src: list[int] = []
    for i in range(boxes_b.shape[0]):
        b = boxes_b[i]
        # Visibility is judged on TRUTH, before any noise -- an object is or is not in view.
        if b[X] <= 0.0:
            continue
        # Deliberately the BEV (2-D) range, not the 3-D ||p_b|| of spec 5.1: a ground-plane
        # tracker gates on ground-plane distance, and the box z is a fixed -EGO_HEIGHT_M here.
        if math.hypot(b[X], b[Y]) >= cfg.max_range_m:
            continue
        if abs(math.atan2(b[Y], b[X])) >= half_fov:
            continue
        if rng.random() >= cfg.p_detect:
            continue
        d = b.copy()
        d[[X, Y, Z]] += rng.normal(0.0, cfg.det_pos_std, 3) if cfg.det_pos_std > 0.0 else 0.0
        if cfg.det_yaw_std > 0.0:
            d[YAW] = float(_wrap(d[YAW] + rng.normal(0.0, cfg.det_yaw_std)))
        dets.append(d)
        src.append(i)

    if cfg.clutter_lambda > 0.0:
        for _ in range(int(rng.poisson(cfg.clutter_lambda))):
            r = rng.uniform(2.0, cfg.max_range_m)
            a = rng.uniform(-half_fov, half_fov)
            c = np.zeros(BOX_DIM)
            c[X] = r * math.cos(a)
            c[Y] = r * math.sin(a)
            c[Z] = -EGO_HEIGHT_M
            c[YAW] = rng.uniform(-math.pi, math.pi)
            c[L], c[W], c[H] = CAR_DIMS
            dets.append(c)
            src.append(-1)

    if not dets:
        return np.zeros((0, BOX_DIM)), np.zeros(0, dtype=int)

    arr = np.asarray(dets, dtype=float)
    ids = np.asarray(src, dtype=int)
    perm = rng.permutation(arr.shape[0])   # a detector emits no identity and no order
    return arr[perm], ids[perm]
