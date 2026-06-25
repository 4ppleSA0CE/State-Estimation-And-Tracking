"""Frame-safe rigid transform utilities for Stage 1 KITTI prototypes."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAP = "map"  # global/world frame name
BASE_LINK = "base_link"  # vehicle body frame name
IMU_LINK = "imu_link"  # IMU sensor frame name
GPS_LINK = "gps_link"  # GPS antenna frame name
VELO_LINK = "velo_link"  # velodyne lidar frame name
_ROTATION_ATOL = 1e-6  # tolerance for orthonormality checks on rotation matrices


class FrameError(RuntimeError):
    """Raised when frame or transform data is malformed or incompatible."""


def _as_float_array(name: str, value: object) -> np.ndarray:
    try:
        return np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise FrameError(f"{name} must be numeric array-like") from exc


def _as_vector3(name: str, value: object) -> np.ndarray:
    array = _as_float_array(name, value)
    if array.shape != (3,):
        raise FrameError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise FrameError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


def _as_rotation(name: str, value: object) -> np.ndarray:
    array = _as_float_array(name, value)
    if array.shape != (3, 3):
        raise FrameError(f"{name} must have shape (3, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise FrameError(f"{name} must contain only finite values")
    if not np.allclose(array.T @ array, np.eye(3), atol=_ROTATION_ATOL, rtol=0.0):
        raise FrameError(f"{name} must be orthonormal")
    determinant = np.linalg.det(array)
    if not np.isclose(determinant, 1.0, atol=_ROTATION_ATOL, rtol=0.0):
        raise FrameError(f"{name} determinant must be +1, got {determinant}")
    array.flags.writeable = False
    return array


def _as_points(name: str, value: object) -> tuple[np.ndarray, bool]:
    array = _as_float_array(name, value)
    single = False

    if array.shape == (3,):
        array = array.reshape(1, 3)
        single = True
    elif array.ndim != 2 or array.shape[1] != 3:
        raise FrameError(f"{name} must have shape (3,) or (N, 3), got {array.shape}")

    if not np.all(np.isfinite(array)):
        raise FrameError(f"{name} must contain only finite values")

    return array, single


def _as_frame_name(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise FrameError(f"{name} must be a non-empty frame name")
    return value


def _as_covariance3(name: str, value: object) -> np.ndarray:
    array = _as_float_array(name, value)
    if array.shape != (3, 3):
        raise FrameError(f"{name} must have shape (3, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise FrameError(f"{name} must contain only finite values")
    if not np.allclose(array, array.T, atol=1e-9, rtol=1e-12):
        raise FrameError(f"{name} must be symmetric")
    return array


def _parse_calibration_line(line: str) -> tuple[str, list[float]] | None:
    stripped = line.strip()
    if not stripped or ":" not in stripped:
        return None

    key, raw_values = stripped.split(":", 1)
    key = key.strip()
    if not key:
        return None

    parts = raw_values.split()
    if not parts:
        return None

    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None

    return key, values


def parse_kitti_rt_file(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    calib_path = Path(path)
    if not calib_path.exists():
        raise FrameError(f"KITTI calibration file does not exist: {calib_path}")

    parsed: dict[str, list[float]] = {}
    for line in calib_path.read_text(encoding="utf-8").splitlines():
        calibration_line = _parse_calibration_line(line)
        if calibration_line is None:
            continue
        key, values = calibration_line
        parsed[key] = values

    for key in ("R", "T"):
        if key not in parsed:
            raise FrameError(f"missing required key {key}")

    if len(parsed["R"]) != 9:
        raise FrameError(f"R must have 9 values, got {len(parsed['R'])}")
    if len(parsed["T"]) != 3:
        raise FrameError(f"T must have 3 values, got {len(parsed['T'])}")

    rotation = _as_rotation("R", np.asarray(parsed["R"], dtype=float).reshape(3, 3))
    translation = _as_vector3("T", parsed["T"])
    return rotation, translation


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform T_target_source mapping coordinates from source to target."""

    target: str  # name of the destination frame, e.g. "map"
    source: str  # name of the source frame, e.g. "imu_link"
    rotation: np.ndarray  # 3x3 R_target_source, rotates vectors from source into target
    translation: np.ndarray  # (3,) translation of source origin expressed in target frame, m

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _as_frame_name("target", self.target))
        object.__setattr__(self, "source", _as_frame_name("source", self.source))
        object.__setattr__(self, "rotation", _as_rotation("rotation", self.rotation))
        object.__setattr__(self, "translation", _as_vector3("translation", self.translation))

    @classmethod
    def identity(cls, frame: str) -> "RigidTransform":
        return cls(
            target=frame,
            source=frame,
            rotation=np.eye(3),
            translation=np.zeros(3),
        )

    def inverse(self) -> "RigidTransform":
        rotation = self.rotation.T
        translation = -(rotation @ self.translation)
        return RigidTransform(
            target=self.source,
            source=self.target,
            rotation=rotation,
            translation=translation,
        )

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        if not isinstance(other, RigidTransform):
            raise FrameError("other must be a RigidTransform")
        if self.source != other.target:
            raise FrameError(
                "cannot compose transforms with mismatched frames: "
                f"{self.target}<-{self.source} after {other.target}<-{other.source}"
            )

        rotation = self.rotation @ other.rotation
        translation = self.rotation @ other.translation + self.translation
        return RigidTransform(
            target=self.target,
            source=other.source,
            rotation=rotation,
            translation=translation,
        )

    def matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def transform_points(self, points: object) -> np.ndarray:
        points_2d, single = _as_points("points", points)
        transformed = points_2d @ self.rotation.T + self.translation
        if single:
            return transformed[0]
        return transformed

    def transform_vectors(self, vectors: object) -> np.ndarray:
        vectors_2d, single = _as_points("vectors", vectors)
        transformed = vectors_2d @ self.rotation.T
        if single:
            return transformed[0]
        return transformed

    def transform_covariance(self, covariance: object) -> np.ndarray:
        covariance_3d = _as_covariance3("covariance", covariance)
        transformed = self.rotation @ covariance_3d @ self.rotation.T
        return 0.5 * (transformed + transformed.T)


@dataclass(frozen=True)
class FramePoint:
    frame: str  # coordinate frame the point is expressed in
    xyz: np.ndarray  # (3,) position vector in that frame, m

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _as_frame_name("point frame", self.frame))
        object.__setattr__(self, "xyz", _as_vector3("point xyz", self.xyz))


@dataclass(frozen=True)
class FrameVector:
    frame: str  # coordinate frame the vector is expressed in
    xyz: np.ndarray  # (3,) free vector (no translation applied when transforming)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _as_frame_name("vector frame", self.frame))
        object.__setattr__(self, "xyz", _as_vector3("vector xyz", self.xyz))


@dataclass(frozen=True)
class FrameCovariance:
    frame: str  # coordinate frame the covariance is expressed in
    covariance: np.ndarray  # (3, 3) symmetric positive-semidefinite covariance matrix

    def __post_init__(self) -> None:
        covariance = _as_covariance3("covariance", self.covariance)
        covariance.flags.writeable = False
        object.__setattr__(self, "frame", _as_frame_name("covariance frame", self.frame))
        object.__setattr__(self, "covariance", covariance)


class Frames:
    """Graph of rigid transforms between named coordinate frames."""

    def __init__(self, transforms: Iterable[RigidTransform]) -> None:
        self._edges: dict[tuple[str, str], RigidTransform] = {}  # adjacency map (target, source) -> transform
        self._frames: set[str] = set()
        for transform in transforms:
            self.add(transform)

    def add(self, transform: RigidTransform) -> None:
        if not isinstance(transform, RigidTransform):
            raise FrameError("transform must be a RigidTransform")

        self._store_edge(transform)
        inverse = transform.inverse()
        self._store_edge(inverse)
        self._frames.update((transform.target, transform.source))

    def _store_edge(self, transform: RigidTransform) -> None:
        key = (transform.target, transform.source)
        existing = self._edges.get(key)
        if existing is None:
            self._edges[key] = transform
            return

        rotations_match = np.allclose(
            existing.rotation,
            transform.rotation,
            atol=1e-12,
            rtol=0.0,
        )
        translations_match = np.allclose(
            existing.translation,
            transform.translation,
            atol=1e-12,
            rtol=0.0,
        )
        if rotations_match and translations_match:
            return

        raise FrameError(
            f"conflicting transform for {transform.target}<-{transform.source}"
        )

    def transform(self, target: str, source: str) -> RigidTransform:
        target = _as_frame_name("target", target)
        source = _as_frame_name("source", source)

        if target == source:
            return RigidTransform.identity(target)

        direct = self._edges.get((target, source))
        if direct is not None:
            return direct

        queue: deque[RigidTransform] = deque([RigidTransform.identity(source)])  # BFS frontier: partial transforms rooted at source
        visited = {source}

        while queue:
            partial = queue.popleft()
            for edge in self._edges.values():
                if edge.source != partial.target or edge.target in visited:
                    continue

                candidate = edge.compose(partial)
                if candidate.target == target:
                    return candidate

                visited.add(candidate.target)
                queue.append(candidate)

        known_frames = ", ".join(sorted(self._frames)) if self._frames else "<none>"
        raise FrameError(
            f"No transform path from {source} to {target}; known frames: {known_frames}"
        )

    def transform_point(self, target: str, point: FramePoint) -> FramePoint:
        if not isinstance(point, FramePoint):
            raise FrameError("point must be a FramePoint")
        transform = self.transform(target, point.frame)
        return FramePoint(transform.target, transform.transform_points(point.xyz))

    def transform_vector(self, target: str, vector: FrameVector) -> FrameVector:
        if not isinstance(vector, FrameVector):
            raise FrameError("vector must be a FrameVector")
        transform = self.transform(target, vector.frame)
        return FrameVector(transform.target, transform.transform_vectors(vector.xyz))

    def transform_covariance(
        self, target: str, covariance: FrameCovariance
    ) -> FrameCovariance:
        if not isinstance(covariance, FrameCovariance):
            raise FrameError("covariance must be a FrameCovariance")
        transform = self.transform(target, covariance.frame)
        return FrameCovariance(
            transform.target,
            transform.transform_covariance(covariance.covariance),
        )


def kitti_imu_to_velo_transform(path: Path | str) -> RigidTransform:
    rotation, translation = parse_kitti_rt_file(path)
    return RigidTransform(
        target=VELO_LINK,
        source=IMU_LINK,
        rotation=rotation,
        translation=translation,
    )


def default_stage1_frames(
    p_base_imu: object = (0.0, 0.0, 0.0),
    p_base_gps: object = (0.0, 0.0, 0.0),
    t_velo_imu: RigidTransform | None = None,
) -> Frames:
    transforms = [
        RigidTransform(
            target=BASE_LINK,
            source=IMU_LINK,
            rotation=np.eye(3),
            translation=p_base_imu,
        ),
        RigidTransform(
            target=BASE_LINK,
            source=GPS_LINK,
            rotation=np.eye(3),
            translation=p_base_gps,
        ),
    ]

    if t_velo_imu is not None:
        if not isinstance(t_velo_imu, RigidTransform):
            raise FrameError("t_velo_imu must be a RigidTransform")
        if t_velo_imu.target != VELO_LINK or t_velo_imu.source != IMU_LINK:
            raise FrameError(
                f"t_velo_imu must be {VELO_LINK}<-{IMU_LINK}, "
                f"got {t_velo_imu.target}<-{t_velo_imu.source}"
            )
        transforms.append(t_velo_imu)

    return Frames(transforms)


def load_kitti_stage1_frames(
    root: Path | str,
    date: str,
    p_base_imu: object = (0.0, 0.0, 0.0),
    p_base_gps: object = (0.0, 0.0, 0.0),
) -> Frames:
    t_velo_imu = kitti_imu_to_velo_transform(
        Path(root) / date / "calib_imu_to_velo.txt"
    )
    return default_stage1_frames(
        p_base_imu=p_base_imu,
        p_base_gps=p_base_gps,
        t_velo_imu=t_velo_imu,
    )
