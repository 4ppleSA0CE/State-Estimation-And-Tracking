"""Scalar-first SO3 and quaternion helpers."""

from __future__ import annotations

import numpy as np


class SO3Error(RuntimeError):
    """Raised when SO3 or quaternion data is malformed."""


def _as_float_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise SO3Error(f"{name} must be numeric array-like") from exc

    if array.shape != shape:
        raise SO3Error(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise SO3Error(f"{name} must contain only finite values")

    return array


def quat_normalize(q: object) -> np.ndarray:
    """Return q as a unit scalar-first quaternion [w, x, y, z]."""
    quat = _as_float_array("q", q, (4,))
    scale = np.max(np.abs(quat))
    if scale <= 0.0:
        raise SO3Error("q must have nonzero norm")

    scaled = quat / scale
    normalized = scaled / np.linalg.norm(scaled)
    if normalized[0] < 0.0:
        normalized = -normalized
    return normalized


def quat_multiply(q_left: object, q_right: object) -> np.ndarray:
    """Return normalized scalar-first rotation composition q_left * q_right."""
    left = quat_normalize(q_left)
    right = quat_normalize(q_right)
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right

    product = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    return quat_normalize(product)


def quat_inverse(q: object) -> np.ndarray:
    """Return the inverse of a unit scalar-first quaternion."""
    w, x, y, z = quat_normalize(q)
    return np.array([w, -x, -y, -z], dtype=float)


def rotvec_to_quat(rotvec: object) -> np.ndarray:
    """Convert a rotation vector to a scalar-first quaternion."""
    vector = _as_float_array("rotvec", rotvec, (3,))
    angle = np.linalg.norm(vector)

    if angle < 1e-12:
        return quat_normalize(np.array([1.0, 0.5 * vector[0], 0.5 * vector[1], 0.5 * vector[2]]))

    axis = vector / angle
    half_angle = 0.5 * angle
    sin_half = np.sin(half_angle)
    return quat_normalize(
        np.array(
            [
                np.cos(half_angle),
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
            ],
            dtype=float,
        )
    )


def quat_to_rotvec(q: object) -> np.ndarray:
    """Convert a scalar-first quaternion to a rotation vector (SO(3) log map)."""
    # log map: inverse of rotvec_to_quat. takes a rotation back to its axis*angle
    # vector. used by the ESKF boxminus to measure an attitude difference.
    w, x, y, z = quat_normalize(q)
    vector = np.array([x, y, z], dtype=float)
    sin_half = np.linalg.norm(vector)  # |[x,y,z]| = sin(angle/2)

    if sin_half < 1e-12:  # near identity, fall back to small-angle: angle ~ 2*[x,y,z]
        return 2.0 * vector

    angle = 2.0 * np.arctan2(sin_half, w)
    return (angle / sin_half) * vector


def quat_to_rotmat(q: object) -> np.ndarray:
    """Convert a scalar-first quaternion to an active 3x3 rotation matrix (quaternion -> DCM, active convention)."""
    w, x, y, z = quat_normalize(q)

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX yaw-pitch-roll Euler angles to a scalar-first quaternion."""
    angles = _as_float_array("euler", [roll, pitch, yaw], (3,))
    half_roll = 0.5 * angles[0]
    half_pitch = 0.5 * angles[1]
    half_yaw = 0.5 * angles[2]

    cr = np.cos(half_roll)   # cos(roll/2)
    sr = np.sin(half_roll)   # sin(roll/2)
    cp = np.cos(half_pitch)  # cos(pitch/2)
    sp = np.sin(half_pitch)  # sin(pitch/2)
    cy = np.cos(half_yaw)    # cos(yaw/2)
    sy = np.sin(half_yaw)    # sin(yaw/2)

    return quat_normalize(
        np.array(
            [
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                sy * cp * sr + cy * sp * cr,
                sy * cp * cr - cy * sp * sr,
            ],
            dtype=float,
        )
    )


def quat_to_euler(q: object) -> np.ndarray:
    """Convert a scalar-first quaternion to ZYX roll, pitch, yaw Euler angles."""
    w, x, y, z = quat_normalize(q)
    rotation = quat_to_rotmat(np.array([w, x, y, z], dtype=float))
    cos_pitch_proxy = np.hypot(rotation[0, 0], rotation[1, 0])  # ~cos(pitch); near zero -> gimbal-lock singularity

    if cos_pitch_proxy <= np.finfo(float).eps:
        sinp = -rotation[2, 0]
        pitch = np.copysign(np.pi / 2.0, sinp)
        roll = 0.0
        if sinp > 0.0:
            yaw = np.arctan2(rotation[1, 2], rotation[0, 2])
        else:
            yaw = np.arctan2(-rotation[1, 2], -rotation[0, 2])
        return np.array([roll, pitch, yaw], dtype=float)

    sinp = -rotation[2, 0]
    pitch = np.arctan2(sinp, cos_pitch_proxy)

    if abs(sinp) >= 1.0 - 8.0 * np.finfo(float).eps:
        # Independent roll/yaw matrix terms are ill-conditioned here; preserve
        # the stable coupled angle while keeping pitch non-singular.
        if sinp > 0.0:
            yaw_roll_coupling = np.arctan2(rotation[1, 2], rotation[0, 2])
            coupling_sign = -1.0
        else:
            yaw_roll_coupling = np.arctan2(-rotation[1, 2], -rotation[0, 2])
            coupling_sign = 1.0

        cos_coupling = np.cos(yaw_roll_coupling)
        sin_coupling = np.sin(yaw_roll_coupling)
        roll_cos_term = rotation[0, 0] * cos_coupling + rotation[1, 0] * sin_coupling + rotation[2, 2]
        roll_sin_term = (
            coupling_sign * rotation[0, 0] * sin_coupling
            - coupling_sign * rotation[1, 0] * cos_coupling
            + rotation[2, 1]
        )
        roll = np.arctan2(roll_sin_term, roll_cos_term)
        yaw = roll + yaw_roll_coupling if sinp > 0.0 else yaw_roll_coupling - roll
        return np.array([roll, pitch, yaw], dtype=float)

    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

    return np.array([roll, pitch, yaw], dtype=float)
