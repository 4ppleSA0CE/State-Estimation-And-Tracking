"""IMU nominal-state mechanization helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kitti_highrate_loader import (
    HighRateOxtsConfig,
    HighRateOxtsSetupError,
    cache_path_for,
    load_highrate_oxts,
    require_highrate_oxts,
)
from so3 import (
    SO3Error,
    euler_to_quat,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    quat_to_euler,
    quat_to_rotmat,
    rotvec_to_quat,
)


GRAVITY_ENU = np.array([0.0, 0.0, -9.80665])
GRAVITY_ENU.setflags(write=False)


class MechanizationError(RuntimeError):
    """Raised when IMU mechanization inputs are malformed."""


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _as_vector3(name: str, value: object) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise MechanizationError(f"{name} must be numeric array-like") from exc

    if array.shape != (3,):
        raise MechanizationError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise MechanizationError(f"{name} must contain only finite values")

    return _readonly(array)


def _as_timestamps(value: object) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise MechanizationError("timestamps must be numeric array-like") from exc

    if array.ndim != 1:
        raise MechanizationError(f"timestamps must have shape (N,), got {array.shape}")
    if array.size < 2:
        raise MechanizationError("timestamps must contain at least two samples")
    if not np.all(np.isfinite(array)):
        raise MechanizationError("timestamps must contain only finite values")
    if not np.all(np.diff(array) > 0.0):
        raise MechanizationError("timestamps must be strictly increasing")

    return _readonly(array)


def _as_samples(name: str, value: object, count: int) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise MechanizationError(f"{name} must be numeric array-like") from exc

    expected_shape = (count, 3)
    if array.shape != expected_shape:
        raise MechanizationError(f"{name} must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise MechanizationError(f"{name} must contain only finite values")

    return _readonly(array)


def _as_positive_dt(value: object) -> float:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise MechanizationError("dt must be a numeric scalar") from exc

    if array.shape != ():
        raise MechanizationError(f"dt must be a scalar, got shape {array.shape}")

    dt = float(array)
    if not np.isfinite(dt) or dt <= 0.0:
        raise MechanizationError("dt must be positive and finite")

    return dt


@dataclass(frozen=True)
class NominalState:
    position: np.ndarray
    velocity: np.ndarray
    q_map_imu: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_vector3("position", self.position))
        object.__setattr__(self, "velocity", _as_vector3("velocity", self.velocity))

        try:
            q_map_imu = quat_normalize(self.q_map_imu)
        except SO3Error as exc:
            raise MechanizationError("q_map_imu must be a finite nonzero quaternion") from exc

        object.__setattr__(self, "q_map_imu", _readonly(np.array(q_map_imu, dtype=float, copy=True)))


@dataclass(frozen=True)
class MechanizationInput:
    timestamps: np.ndarray
    accel_body: np.ndarray
    gyro_body: np.ndarray

    def __post_init__(self) -> None:
        timestamps = _as_timestamps(self.timestamps)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "accel_body", _as_samples("accel_body", self.accel_body, timestamps.size))
        object.__setattr__(self, "gyro_body", _as_samples("gyro_body", self.gyro_body, timestamps.size))


@dataclass(frozen=True)
class MechanizationResult:
    timestamps: np.ndarray
    states: tuple[NominalState, ...]

    def __post_init__(self) -> None:
        timestamps = _as_timestamps(self.timestamps)
        try:
            states = tuple(self.states)
        except TypeError as exc:
            raise MechanizationError("states must be iterable") from exc

        if len(states) != timestamps.size:
            raise MechanizationError("states must contain one state for every timestamp")
        if not all(isinstance(state, NominalState) for state in states):
            raise MechanizationError("states must contain only NominalState values")

        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "states", states)

    @property
    def final_state(self) -> NominalState:
        return self.states[-1]


def _propagate_state_validated(
    state: NominalState,
    accel_body: object,
    gyro_body: object,
    dt: object,
    accel_bias: np.ndarray,
    gyro_bias: np.ndarray,
) -> NominalState:
    if not isinstance(state, NominalState):
        raise MechanizationError("state must be a NominalState")

    dt_seconds = _as_positive_dt(dt)
    accel = _as_vector3("accel_body", accel_body) - accel_bias
    gyro = _as_vector3("gyro_body", gyro_body) - gyro_bias

    rotation = quat_to_rotmat(state.q_map_imu)
    accel_map = rotation @ accel + GRAVITY_ENU

    position = state.position + state.velocity * dt_seconds + 0.5 * accel_map * dt_seconds * dt_seconds
    velocity = state.velocity + accel_map * dt_seconds
    delta_q = rotvec_to_quat(gyro * dt_seconds)
    q_map_imu = quat_multiply(state.q_map_imu, delta_q)

    return NominalState(position=position, velocity=velocity, q_map_imu=q_map_imu)


def propagate_state(
    state: NominalState,
    accel_body: object,
    gyro_body: object,
    dt: object,
    accel_bias: object = (0.0, 0.0, 0.0),
    gyro_bias: object = (0.0, 0.0, 0.0),
) -> NominalState:
    return _propagate_state_validated(
        state=state,
        accel_body=accel_body,
        gyro_body=gyro_body,
        dt=dt,
        accel_bias=_as_vector3("accel_bias", accel_bias),
        gyro_bias=_as_vector3("gyro_bias", gyro_bias),
    )


def mechanize(
    samples: MechanizationInput,
    initial_state: NominalState,
    accel_bias: object = (0.0, 0.0, 0.0),
    gyro_bias: object = (0.0, 0.0, 0.0),
) -> MechanizationResult:
    if not isinstance(samples, MechanizationInput):
        raise MechanizationError("samples must be a MechanizationInput")
    if not isinstance(initial_state, NominalState):
        raise MechanizationError("initial_state must be a NominalState")

    accel_bias_vector = _as_vector3("accel_bias", accel_bias)
    gyro_bias_vector = _as_vector3("gyro_bias", gyro_bias)

    states = [initial_state]
    state = initial_state
    for index, dt in enumerate(np.diff(samples.timestamps)):
        state = _propagate_state_validated(
            state=state,
            accel_body=samples.accel_body[index],
            gyro_body=samples.gyro_body[index],
            dt=dt,
            accel_bias=accel_bias_vector,
            gyro_bias=gyro_bias_vector,
        )
        states.append(state)

    return MechanizationResult(timestamps=samples.timestamps, states=tuple(states))


def select_window(samples: MechanizationInput, duration_s: object) -> MechanizationInput:
    if not isinstance(samples, MechanizationInput):
        raise MechanizationError("samples must be a MechanizationInput")

    try:
        duration_array = np.array(duration_s, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise MechanizationError("duration_s must be a positive finite scalar") from exc
    if duration_array.shape != ():
        raise MechanizationError(f"duration_s must be a scalar, got shape {duration_array.shape}")

    duration = float(duration_array)
    if not np.isfinite(duration) or duration <= 0.0:
        raise MechanizationError("duration_s must be positive and finite")

    end_time = samples.timestamps[0] + duration
    count = int(np.searchsorted(samples.timestamps, end_time, side="right"))
    count = min(max(count, 2), samples.timestamps.size)
    return MechanizationInput(
        timestamps=samples.timestamps[:count],
        accel_body=samples.accel_body[:count],
        gyro_body=samples.gyro_body[:count],
    )


def position_error_m(estimated: NominalState, reference: NominalState) -> float:
    if not isinstance(estimated, NominalState):
        raise MechanizationError("estimated must be a NominalState")
    if not isinstance(reference, NominalState):
        raise MechanizationError("reference must be a NominalState")

    return float(np.linalg.norm(estimated.position - reference.position))


def attitude_error_deg(estimated: NominalState, reference: NominalState) -> float:
    if not isinstance(estimated, NominalState):
        raise MechanizationError("estimated must be a NominalState")
    if not isinstance(reference, NominalState):
        raise MechanizationError("reference must be a NominalState")

    delta = quat_multiply(quat_inverse(reference.q_map_imu), estimated.q_map_imu)
    angle_rad = 2.0 * np.arctan2(np.linalg.norm(delta[1:]), abs(float(delta[0])))
    return float(np.degrees(angle_rad))


def _oxts_field(sequence: object, name: str) -> object:
    try:
        return getattr(sequence, name)
    except AttributeError as exc:
        raise MechanizationError(f"OXTS sequence must provide {name}") from exc


def _oxts_indexed_field(sequence: object, name: str, index: int) -> object:
    values = _oxts_field(sequence, name)
    try:
        return values[index]
    except (TypeError, IndexError, KeyError) as exc:
        raise MechanizationError(f"index {index!r} is out of range for OXTS sequence field {name}") from exc


def initial_state_from_oxts(sequence: object, index: int = 0) -> NominalState:
    roll_pitch_yaw = _as_vector3(
        f"OXTS sequence roll_pitch_yaw[{index}]",
        _oxts_indexed_field(sequence, "roll_pitch_yaw", index),
    )
    position = _as_vector3(
        f"OXTS sequence enu_position_m[{index}]",
        _oxts_indexed_field(sequence, "enu_position_m", index),
    )
    velocity_body = _as_vector3(
        f"OXTS sequence velocity[{index}]",
        _oxts_indexed_field(sequence, "velocity", index),
    )
    try:
        q_map_imu = euler_to_quat(*roll_pitch_yaw)
    except SO3Error as exc:
        raise MechanizationError(f"OXTS sequence roll_pitch_yaw[{index}] must define a valid attitude") from exc

    return NominalState(
        position=position,
        velocity=quat_to_rotmat(q_map_imu) @ velocity_body,
        q_map_imu=q_map_imu,
    )


def mechanization_input_from_oxts(sequence: object) -> MechanizationInput:
    return MechanizationInput(
        timestamps=_oxts_field(sequence, "timestamps"),
        accel_body=_oxts_field(sequence, "accel_body"),
        gyro_body=_oxts_field(sequence, "gyro_body"),
    )


def format_mechanization_summary(
    config: HighRateOxtsConfig,
    result: MechanizationResult,
    reference_final: NominalState,
    cache_path: Path | str,
    plot_path: Path | str | None,
) -> str:
    if not isinstance(config, HighRateOxtsConfig):
        raise MechanizationError("config must be a HighRateOxtsConfig")
    if not isinstance(result, MechanizationResult):
        raise MechanizationError("result must be a MechanizationResult")
    if not isinstance(reference_final, NominalState):
        raise MechanizationError("reference_final must be a NominalState")

    duration_s = float(result.timestamps[-1] - result.timestamps[0])
    imu_rate_hz = float((result.timestamps.size - 1) / duration_s)
    lines = [
        f"sequence: {config.date} drive {config.normalized_drive()} extract",
        f"samples: {result.timestamps.size}",
        f"imu_rate_hz: {imu_rate_hz:.3f}",
        f"duration_s: {duration_s:.3f}",
        f"position_drift_m: {position_error_m(result.final_state, reference_final):.3f}",
        f"attitude_drift_deg: {attitude_error_deg(result.final_state, reference_final):.3f}",
        f"cache: {Path(cache_path)}",
    ]
    if plot_path is not None:
        lines.append(f"plot: {Path(plot_path)}")
    return "\n".join(lines)


def default_plot_path(date: str, drive: str) -> Path:
    return Path("prototypes/output") / f"kitti_{date}_{drive.zfill(4)}_mechanization_drift.png"


def plot_mechanization_xy(
    result: MechanizationResult,
    reference_positions: object,
    output_path: Path | str,
) -> Path:
    if not isinstance(result, MechanizationResult):
        raise MechanizationError("result must be a MechanizationResult")

    reference = _as_samples("reference_positions", reference_positions, result.timestamps.size)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    estimated = np.asarray([state.position for state in result.states], dtype=float)
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(reference[:, 0], reference[:, 1], label="KITTI OXTS reference", linewidth=2.0)
    axis.plot(estimated[:, 0], estimated[:, 1], label="Mechanized IMU", linewidth=2.0)
    axis.set_xlabel("East [m]")
    axis.set_ylabel("North [m]")
    axis.set_title("KITTI IMU Mechanization XY Drift")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mechanize high-rate KITTI Raw OXTS IMU samples.")
    parser.add_argument("--root", type=Path, default=HighRateOxtsConfig.root)
    parser.add_argument("--date", default=HighRateOxtsConfig.date)
    parser.add_argument("--drive", default=HighRateOxtsConfig.drive)
    parser.add_argument("--cache-root", type=Path, default=HighRateOxtsConfig.cache_root)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--plot", nargs="?", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = HighRateOxtsConfig(
        root=args.root,
        date=args.date,
        drive=args.drive,
        cache_root=args.cache_root,
        force_refresh=args.force_refresh,
    )

    try:
        require_highrate_oxts(config)
        sequence = load_highrate_oxts(config)
        all_samples = mechanization_input_from_oxts(sequence)
        samples = select_window(all_samples, args.duration)
        initial_state = initial_state_from_oxts(sequence)
        result = mechanize(samples, initial_state)
        final_index = samples.timestamps.size - 1
        reference_final = initial_state_from_oxts(sequence, index=final_index)

        plot_path = None
        if not args.no_plot:
            output_path = args.plot if args.plot is not None else default_plot_path(config.date, config.drive)
            plot_path = plot_mechanization_xy(result, sequence.enu_position_m[: samples.timestamps.size], output_path)

        print(
            format_mechanization_summary(
                config=config,
                result=result,
                reference_final=reference_final,
                cache_path=cache_path_for(config),
                plot_path=plot_path,
            )
        )
        return 0
    except (HighRateOxtsSetupError, MechanizationError) as exc:
        print(f"error: {exc}")
        return 2


__all__ = [
    "GRAVITY_ENU",
    "MechanizationError",
    "MechanizationInput",
    "MechanizationResult",
    "NominalState",
    "attitude_error_deg",
    "default_plot_path",
    "format_mechanization_summary",
    "initial_state_from_oxts",
    "main",
    "mechanize",
    "mechanization_input_from_oxts",
    "parse_args",
    "plot_mechanization_xy",
    "position_error_m",
    "propagate_state",
    "select_window",
]


if __name__ == "__main__":
    raise SystemExit(main())
