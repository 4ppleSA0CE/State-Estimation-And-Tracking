from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from imu_mechanization import (
    GRAVITY_ENU,
    MechanizationError,
    MechanizationInput,
    MechanizationResult,
    NominalState,
    attitude_error_deg,
    default_plot_path,
    format_mechanization_summary,
    initial_state_from_oxts,
    main,
    mechanize,
    mechanization_input_from_oxts,
    parse_args,
    plot_mechanization_xy,
    position_error_m,
    propagate_state,
    select_window,
)
from kitti_highrate_loader import HighRateOxtsConfig, HighRateOxtsSequence, cache_path_for, save_cache
from so3 import euler_to_quat, quat_multiply, quat_to_euler, quat_to_rotmat, rotvec_to_quat


def _initial_state():
    return NominalState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        q_map_imu=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def _stationary_accel_body(count):
    return np.tile(np.array([0.0, 0.0, -GRAVITY_ENU[2]], dtype=float), (count, 1))


def _oxts_sequence():
    return SimpleNamespace(
        timestamps=np.array([0.0, 0.1, 0.2]),
        accel_body=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]]),
        gyro_body=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        enu_position_m=np.array([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        velocity=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.5], [0.7, 0.8, 0.9]]),
        roll_pitch_yaw=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2.0], [0.4, 0.5, 0.6]]),
    )


def test_gravity_constant_is_read_only():
    assert not GRAVITY_ENU.flags.writeable

    with pytest.raises(ValueError):
        GRAVITY_ENU[2] = 0.0


def test_stationary_identity_attitude_stays_fixed():
    timestamps = np.linspace(0.0, 1.0, 11)
    samples = MechanizationInput(
        timestamps=timestamps,
        accel_body=_stationary_accel_body(timestamps.size),
        gyro_body=np.zeros((timestamps.size, 3)),
    )

    result = mechanize(samples, _initial_state())

    np.testing.assert_allclose(result.final_state.position, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(result.final_state.velocity, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(result.final_state.q_map_imu, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-12)


def test_constant_yaw_rate_integrates_body_gyro():
    timestamps = np.linspace(0.0, 1.0, 101)
    samples = MechanizationInput(
        timestamps=timestamps,
        accel_body=_stationary_accel_body(timestamps.size),
        gyro_body=np.tile(np.array([0.0, 0.0, np.pi / 2.0], dtype=float), (timestamps.size, 1)),
    )

    result = mechanize(samples, _initial_state())
    roll, pitch, yaw = quat_to_euler(result.final_state.q_map_imu)

    assert roll == pytest.approx(0.0, abs=1e-12)
    assert pitch == pytest.approx(0.0, abs=1e-12)
    assert yaw == pytest.approx(np.pi / 2.0, abs=1e-12)


def test_tilted_stationary_attitude_uses_map_from_body_rotation_and_enu_gravity():
    q_map_imu = euler_to_quat(0.3, -0.2, 0.4)
    specific_force_body = -(quat_to_rotmat(q_map_imu).T @ GRAVITY_ENU)
    timestamps = np.linspace(0.0, 1.0, 11)
    samples = MechanizationInput(
        timestamps=timestamps,
        accel_body=np.tile(specific_force_body, (timestamps.size, 1)),
        gyro_body=np.zeros((timestamps.size, 3)),
    )
    initial_state = NominalState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        q_map_imu=q_map_imu,
    )

    result = mechanize(samples, initial_state)

    np.testing.assert_allclose(result.final_state.position, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(result.final_state.velocity, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(result.final_state.q_map_imu, q_map_imu, atol=1e-12)


def test_non_identity_attitude_right_multiplies_body_gyro_increment():
    initial = NominalState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        q_map_imu=euler_to_quat(0.2, -0.3, 0.4),
    )
    delta = rotvec_to_quat([0.1, -0.05, 0.2])

    propagated = propagate_state(
        initial,
        accel_body=np.array([0.0, 0.0, -GRAVITY_ENU[2]]),
        gyro_body=np.array([0.1, -0.05, 0.2]),
        dt=1.0,
    )

    np.testing.assert_allclose(propagated.q_map_imu, quat_multiply(initial.q_map_imu, delta), atol=1e-12)


def test_constant_body_x_acceleration_integrates_position_and_velocity():
    timestamps = np.linspace(0.0, 1.0, 11)
    samples = MechanizationInput(
        timestamps=timestamps,
        accel_body=np.tile(np.array([1.0, 0.0, -GRAVITY_ENU[2]], dtype=float), (timestamps.size, 1)),
        gyro_body=np.zeros((timestamps.size, 3)),
    )

    result = mechanize(samples, _initial_state())

    np.testing.assert_allclose(result.final_state.velocity, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(result.final_state.position, np.array([0.5, 0.0, 0.0]), atol=1e-12)


def test_fixed_bias_inputs_are_subtracted_before_propagation():
    timestamps = np.linspace(0.0, 1.0, 11)
    samples = MechanizationInput(
        timestamps=timestamps,
        accel_body=np.tile(np.array([1.5, 0.0, -GRAVITY_ENU[2]], dtype=float), (timestamps.size, 1)),
        gyro_body=np.tile(np.array([0.0, 0.0, 0.2], dtype=float), (timestamps.size, 1)),
    )

    result = mechanize(
        samples,
        _initial_state(),
        accel_bias=np.array([0.5, 0.0, 0.0]),
        gyro_bias=np.array([0.0, 0.0, 0.2]),
    )
    roll, pitch, yaw = quat_to_euler(result.final_state.q_map_imu)

    np.testing.assert_allclose(result.final_state.velocity, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert roll == pytest.approx(0.0, abs=1e-12)
    assert pitch == pytest.approx(0.0, abs=1e-12)
    assert yaw == pytest.approx(0.0, abs=1e-12)


def test_invalid_timestamps_and_sample_shapes_raise_mechanization_error():
    with pytest.raises(MechanizationError, match="strictly increasing"):
        MechanizationInput(
            timestamps=np.array([0.0, 0.0]),
            accel_body=np.zeros((2, 3)),
            gyro_body=np.zeros((2, 3)),
        )

    with pytest.raises(MechanizationError, match="shape"):
        MechanizationInput(
            timestamps=np.array([0.0, 1.0]),
            accel_body=np.zeros((2, 2)),
            gyro_body=np.zeros((2, 3)),
        )

    with pytest.raises(MechanizationError, match="at least two"):
        MechanizationInput(
            timestamps=np.array([0.0]),
            accel_body=np.zeros((1, 3)),
            gyro_body=np.zeros((1, 3)),
        )


def test_mechanization_result_rejects_non_iterable_states_with_mechanization_error():
    with pytest.raises(MechanizationError, match="states"):
        MechanizationResult(timestamps=np.array([0.0, 1.0]), states=None)


def test_position_and_attitude_error_metrics():
    estimated = NominalState(
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.zeros(3),
        q_map_imu=euler_to_quat(0.0, 0.0, np.deg2rad(10.0)),
    )
    reference = NominalState(
        position=np.array([1.0, 4.0, 3.0]),
        velocity=np.zeros(3),
        q_map_imu=euler_to_quat(0.0, 0.0, np.deg2rad(25.0)),
    )

    assert position_error_m(estimated, reference) == pytest.approx(2.0)
    assert attitude_error_deg(estimated, reference) == pytest.approx(15.0)


def test_select_window_keeps_samples_through_requested_duration():
    samples = MechanizationInput(
        timestamps=np.array([0.0, 0.1, 0.2, 0.3]),
        accel_body=np.zeros((4, 3)),
        gyro_body=np.zeros((4, 3)),
    )

    selected = select_window(samples, duration_s=0.2)

    np.testing.assert_allclose(selected.timestamps, np.array([0.0, 0.1, 0.2]))
    assert selected.accel_body.shape == (3, 3)
    assert selected.gyro_body.shape == (3, 3)

    with pytest.raises(MechanizationError, match="duration"):
        select_window(samples, duration_s=0.0)


def test_oxts_sequence_helpers_create_initial_state_and_mechanization_input():
    sequence = _oxts_sequence()

    state = initial_state_from_oxts(sequence, index=1)
    samples = mechanization_input_from_oxts(sequence)
    expected_q_map_imu = euler_to_quat(0.0, 0.0, np.pi / 2.0)
    expected_velocity = quat_to_rotmat(expected_q_map_imu) @ sequence.velocity[1]

    np.testing.assert_allclose(state.position, np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(state.velocity, expected_velocity, atol=1e-12)
    np.testing.assert_allclose(state.q_map_imu, expected_q_map_imu)
    np.testing.assert_allclose(samples.timestamps, sequence.timestamps)
    np.testing.assert_allclose(samples.accel_body, sequence.accel_body)
    np.testing.assert_allclose(samples.gyro_body, sequence.gyro_body)


def test_initial_state_from_oxts_wraps_malformed_sequence_errors():
    with pytest.raises(MechanizationError, match="OXTS sequence.*roll_pitch_yaw"):
        initial_state_from_oxts(SimpleNamespace())


def test_initial_state_from_oxts_wraps_bad_indices():
    with pytest.raises(MechanizationError, match="index.*out of range"):
        initial_state_from_oxts(_oxts_sequence(), index=10)


def test_mechanization_input_from_oxts_wraps_malformed_sequence_errors():
    with pytest.raises(MechanizationError, match="OXTS sequence.*accel_body"):
        mechanization_input_from_oxts(SimpleNamespace(timestamps=np.array([0.0, 0.1])))


def test_plot_mechanization_xy_rejects_nonnumeric_reference_positions(tmp_path):
    result = MechanizationResult(
        timestamps=np.array([0.0, 0.1]),
        states=(_initial_state(), _initial_state()),
    )

    with pytest.raises(MechanizationError, match="reference_positions.*numeric"):
        plot_mechanization_xy(result, [["bad", 0.0, 0.0], [1.0, 0.0, 0.0]], tmp_path / "xy.png")


def test_plot_mechanization_xy_rejects_nonfinite_reference_positions(tmp_path):
    result = MechanizationResult(
        timestamps=np.array([0.0, 0.1]),
        states=(_initial_state(), _initial_state()),
    )
    reference = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0]])

    with pytest.raises(MechanizationError, match="reference_positions.*finite"):
        plot_mechanization_xy(result, reference, tmp_path / "xy.png")


def test_format_mechanization_summary_includes_required_lines(tmp_path):
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="1",
        cache_root=tmp_path / "cache",
    )
    result = MechanizationResult(
        timestamps=np.array([0.0, 0.1]),
        states=(
            NominalState(position=np.zeros(3), velocity=np.zeros(3), q_map_imu=euler_to_quat(0.0, 0.0, 0.0)),
            NominalState(
                position=np.array([3.0, 4.0, 0.0]),
                velocity=np.zeros(3),
                q_map_imu=euler_to_quat(0.0, 0.0, np.deg2rad(20.0)),
            ),
        ),
    )
    reference_final = NominalState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        q_map_imu=euler_to_quat(0.0, 0.0, np.deg2rad(5.0)),
    )
    cache_path = tmp_path / "cache" / "sequence.npz"
    plot_path = tmp_path / "plots" / "xy.png"

    summary = format_mechanization_summary(
        config=config,
        result=result,
        reference_final=reference_final,
        cache_path=cache_path,
        plot_path=plot_path,
    )

    assert "sequence: 2011_09_26 drive 0001 extract" in summary
    assert "samples: 2" in summary
    assert "imu_rate_hz: 10.000" in summary
    assert "duration_s: 0.100" in summary
    assert "position_drift_m: 5.000" in summary
    assert "attitude_drift_deg: 15.000" in summary
    assert f"cache: {cache_path}" in summary
    assert f"plot: {plot_path}" in summary


def test_default_plot_path_uses_date_drive():
    assert default_plot_path("2011_09_26", "1") == Path(
        "prototypes/output/kitti_2011_09_26_0001_mechanization_drift.png"
    )


def test_parse_args_defaults_and_no_plot():
    args = parse_args(["--duration", "2.5", "--no-plot"])

    assert args.duration == pytest.approx(2.5)
    assert args.no_plot is True
    assert args.root == HighRateOxtsConfig.root
    assert args.date == HighRateOxtsConfig.date
    assert args.drive == HighRateOxtsConfig.drive


def test_main_rejects_sync_only_drive_even_when_matching_cache_exists(tmp_path, capsys):
    root = tmp_path / "kitti_raw"
    cache_root = tmp_path / "cache"
    date = "2011_09_26"
    drive = "0001"
    config = HighRateOxtsConfig(root=root, date=date, drive=drive, cache_root=cache_root)

    sync_oxts_path = root / date / f"{date}_drive_{drive}_sync" / "oxts"
    sync_data_path = sync_oxts_path / "data"
    sync_data_path.mkdir(parents=True)
    (sync_oxts_path / "timestamps.txt").write_text(
        "2011-09-26 13:02:25.000000000\n2011-09-26 13:02:25.100000000\n",
        encoding="utf-8",
    )
    (sync_data_path / "0000000000.txt").write_text("0\n", encoding="utf-8")

    cached_sequence = HighRateOxtsSequence(
        timestamps=np.array([0.0, 0.1]),
        lat_lon_alt=np.array([[49.0, 8.0, 100.0], [49.0, 8.0, 100.0]]),
        enu_position_m=np.zeros((2, 3)),
        roll_pitch_yaw=np.zeros((2, 3)),
        velocity=np.zeros((2, 3)),
        accel_body=np.tile(np.array([0.0, 0.0, -GRAVITY_ENU[2]]), (2, 1)),
        gyro_body=np.zeros((2, 3)),
        origin_lat_lon_alt=np.array([49.0, 8.0, 100.0]),
        date=date,
        drive=drive,
        source_path=str(root / date / f"{date}_drive_{drive}_extract" / "oxts"),
    )
    save_cache(cached_sequence, cache_path_for(config))

    status = main(
        [
            "--root",
            str(root),
            "--date",
            date,
            "--drive",
            drive,
            "--cache-root",
            str(cache_root),
            "--no-plot",
        ]
    )

    output = capsys.readouterr().out
    assert status == 2
    assert "high-rate" in output.lower()
    assert "extract" in output.lower() or "synced" in output.lower()
