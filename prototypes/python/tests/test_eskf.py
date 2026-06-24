import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from eskf import EskfError, ErrorStateEKF, EskfConfig, boxminus, boxplus, skew, build_gps_measurements, nees_per_step, position_rmse, run_eskf, default_plot_path, parse_args, plot_eskf_summary
from imu_mechanization import GRAVITY_ENU, NominalState, propagate_state
from kitti_highrate_loader import HighRateOxtsConfig, HighRateOxtsSetupError, load_highrate_oxts, require_highrate_oxts
from so3 import euler_to_quat, quat_to_rotmat


def _nominal():
    return NominalState(
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([0.5, -0.5, 0.1]),
        q_map_imu=euler_to_quat(0.1, -0.2, 0.3),
    )


def test_skew_matches_cross_product():
    v = np.array([0.3, -0.7, 1.1])
    w = np.array([2.0, 1.0, -0.5])
    np.testing.assert_allclose(skew(v) @ w, np.cross(v, w), atol=1e-15)


def test_skew_is_antisymmetric():
    s = skew([1.0, 2.0, 3.0])
    np.testing.assert_allclose(s, -s.T, atol=1e-15)


def test_skew_rejects_bad_shape():
    with pytest.raises(EskfError, match="shape"):
        skew([1.0, 2.0])


def test_boxplus_then_boxminus_round_trips():
    nominal = _nominal()
    accel_bias = np.array([0.01, -0.02, 0.03])
    gyro_bias = np.array([0.001, 0.002, -0.003])
    dx = np.array([0.1, -0.2, 0.3, 0.05, -0.05, 0.02, -0.01, 0.03, 0.04, 0.001, -0.002, 0.003, 0.0, 0.0, 0.0])

    nom2, ba2, bg2 = boxplus(nominal, accel_bias, gyro_bias, dx)
    recovered = boxminus(nom2, ba2, bg2, nominal, accel_bias, gyro_bias)

    np.testing.assert_allclose(recovered, dx, atol=1e-9)


def test_boxplus_then_boxminus_round_trips_at_large_angle():
    nominal = _nominal()
    accel_bias = np.array([0.01, -0.02, 0.03])
    gyro_bias = np.array([0.001, 0.002, -0.003])
    dx = np.array([1.0, -2.0, 3.0, 0.5, -0.5, 0.2, 0.6, -0.5, 0.7, 0.01, -0.02, 0.03, 0.0, 0.0, 0.0])

    nom2, ba2, bg2 = boxplus(nominal, accel_bias, gyro_bias, dx)
    recovered = boxminus(nom2, ba2, bg2, nominal, accel_bias, gyro_bias)

    np.testing.assert_allclose(recovered, dx, atol=1e-9)


def test_boxminus_zero_for_identical_states():
    nominal = _nominal()
    ba = np.zeros(3)
    bg = np.zeros(3)
    np.testing.assert_allclose(boxminus(nominal, ba, bg, nominal, ba, bg), np.zeros(15), atol=1e-12)


def _filter():
    nominal = _nominal()
    config = EskfConfig()
    return ErrorStateEKF(nominal, config, accel_bias=np.zeros(3), gyro_bias=np.zeros(3))


def _assert_symmetric_psd(matrix, tol=1e-9):
    np.testing.assert_allclose(matrix, matrix.T, atol=tol)
    assert np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))) >= -tol


def test_predict_keeps_covariance_symmetric_psd():
    filt = _filter()
    accel = np.array([0.2, -0.1, 9.9])
    gyro = np.array([0.01, -0.02, 0.03])
    for _ in range(50):
        filt.predict(accel, gyro, dt=0.01)
    _assert_symmetric_psd(filt.P)
    assert filt.P.shape == (15, 15)


def test_predict_jacobian_matches_finite_difference():
    nominal = _nominal()
    accel_bias = np.array([0.02, -0.01, 0.03])
    gyro_bias = np.array([0.001, -0.002, 0.003])
    accel = np.array([0.3, -0.2, 9.7])
    gyro = np.array([0.05, -0.04, 0.06])
    dt = 0.01

    filt = ErrorStateEKF(nominal, EskfConfig(), accel_bias=accel_bias, gyro_bias=gyro_bias)
    analytic = filt._state_transition(accel, gyro, dt)

    base = propagate_state(nominal, accel, gyro, dt, accel_bias, gyro_bias)
    eps = 1e-6
    numeric = np.zeros((15, 15))
    for i in range(15):
        perturb = np.zeros(15)
        perturb[i] = eps
        nom_p, ba_p, bg_p = boxplus(nominal, accel_bias, gyro_bias, perturb)
        prop_p = propagate_state(nom_p, accel, gyro, dt, ba_p, bg_p)
        numeric[:, i] = boxminus(prop_p, ba_p, bg_p, base, accel_bias, gyro_bias) / eps

    # atol=1e-5 absorbs the only unmodelled term: the attitude/gyro-bias right-Jacobian
    # correction (~3e-6 at dt=0.01), which the standard first-order -R*dt block omits.
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-5)


def test_gps_update_jacobian_matches_finite_difference_with_lever_arm():
    nominal = _nominal()
    config = EskfConfig(p_base_gps=(0.3, -0.2, 0.1))
    filt = ErrorStateEKF(nominal, config)
    analytic = filt._measurement_jacobian()

    def h(nom):
        return nom.position + quat_to_rotmat(nom.q_map_imu) @ np.array([0.3, -0.2, 0.1])

    eps = 1e-6
    numeric = np.zeros((3, 15))
    for i in range(15):
        perturb = np.zeros(15)
        perturb[i] = eps
        nom_p, _, _ = boxplus(nominal, np.zeros(3), np.zeros(3), perturb)
        numeric[:, i] = (h(nom_p) - h(nominal)) / eps

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_gps_update_keeps_covariance_symmetric_psd():
    filt = _filter()
    filt.update_gps(filt.nominal.position + np.array([0.5, -0.3, 0.1]))
    _assert_symmetric_psd(filt.P)


def test_gps_update_moves_position_toward_measurement():
    filt = _filter()
    target = filt.nominal.position + np.array([3.0, -2.0, 0.0])
    start_err = np.linalg.norm(filt.nominal.position - target)
    filt.update_gps(target)
    end_err = np.linalg.norm(filt.nominal.position - target)
    assert end_err < start_err


def test_stationary_filter_converges_to_gps_truth():
    truth = np.array([10.0, -5.0, 2.0])
    nominal = NominalState(
        position=truth + np.array([3.0, -3.0, 1.0]),
        velocity=np.zeros(3),
        q_map_imu=euler_to_quat(0.0, 0.0, 0.0),
    )
    filt = ErrorStateEKF(nominal, EskfConfig())
    specific_force = np.array([0.0, 0.0, -GRAVITY_ENU[2]])  # stationary, identity attitude
    rng = np.random.default_rng(0)
    trace_start = np.trace(filt.P[0:3, 0:3])
    for k in range(2000):
        filt.predict(specific_force, np.zeros(3), dt=0.01)
        if k % 10 == 0:
            filt.update_gps(truth + rng.normal(0.0, 1.5, size=3))
    assert np.linalg.norm(filt.nominal.position - truth) < 1.0
    assert np.trace(filt.P[0:3, 0:3]) < trace_start


def _synthetic_sequence(n=120):
    dt = 0.01
    timestamps = np.arange(n) * dt
    # Straight constant-velocity drive at 8 m/s east, identity attitude.
    enu = np.zeros((n, 3))
    enu[:, 0] = 8.0 * timestamps
    velocity_body = np.tile(np.array([8.0, 0.0, 0.0]), (n, 1))
    roll_pitch_yaw = np.zeros((n, 3))
    accel_body = np.tile(np.array([0.0, 0.0, -GRAVITY_ENU[2]]), (n, 1))
    gyro_body = np.zeros((n, 3))
    return SimpleNamespace(
        timestamps=timestamps,
        enu_position_m=enu,
        velocity=velocity_body,
        roll_pitch_yaw=roll_pitch_yaw,
        accel_body=accel_body,
        gyro_body=gyro_body,
    )


def test_build_gps_measurements_subsamples_and_adds_noise():
    enu = np.zeros((100, 3))
    enu[:, 0] = np.arange(100)
    indices, z = build_gps_measurements(enu, divisor=10, gps_std_m=1.5, seed=0)
    assert indices.tolist() == list(range(0, 100, 10))
    assert z.shape == (10, 3)
    assert np.all(np.abs(z[:, 0] - enu[indices, 0]) < 10.0)


def test_run_eskf_is_deterministic():
    seq = _synthetic_sequence()
    first = run_eskf(seq, EskfConfig(), seed=7)
    second = run_eskf(seq, EskfConfig(), seed=7)
    np.testing.assert_allclose(first["x_est"], second["x_est"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first["nees"], second["nees"], rtol=0.0, atol=0.0)


def test_run_eskf_tracks_synthetic_truth():
    seq = _synthetic_sequence(n=300)
    result = run_eskf(seq, EskfConfig(), seed=1)
    assert result["x_est"].shape == (300, 3)
    assert result["position_rmse"] < 1.5  # within GPS noise on a clean synthetic run
    assert np.isfinite(result["attitude_rmse_deg"])
    assert np.all(np.isfinite(result["nees"]))


def test_parse_args_defaults():
    args = parse_args(["--duration", "5.0", "--no-plot"])
    assert args.duration == 5.0
    assert args.no_plot is True
    assert args.date == HighRateOxtsConfig.date


def test_default_plot_path_uses_date_drive():
    assert default_plot_path("2011_09_26", "1") == Path(
        "prototypes/output/kitti_2011_09_26_0001_eskf_summary.png"
    )


def test_plot_eskf_summary_writes_png(tmp_path):
    seq = _synthetic_sequence(n=200)
    result = run_eskf(seq, EskfConfig(), seed=2)
    out = tmp_path / "eskf.png"
    plot_eskf_summary(result, EskfConfig(), out)
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_run_eskf_dropout_increases_error_in_window():
    seq = _synthetic_sequence(n=400)
    clean = run_eskf(seq, EskfConfig(), seed=3)
    dropped = run_eskf(seq, EskfConfig(), seed=3, dropout_window=(1.0, 3.0))
    t = clean["timestamps"]
    mask = (t >= 1.0) & (t <= 3.0)
    clean_err = np.linalg.norm(clean["x_est"][mask, 0:2] - clean["truth_positions"][mask, 0:2], axis=1)
    drop_err = np.linalg.norm(dropped["x_est"][mask, 0:2] - dropped["truth_positions"][mask, 0:2], axis=1)
    assert drop_err.mean() > clean_err.mean()


def _extract_available() -> bool:
    config = HighRateOxtsConfig()
    try:
        require_highrate_oxts(config)
        return True
    except HighRateOxtsSetupError:
        return False


@pytest.mark.skipif(not _extract_available(), reason="local KITTI Raw extract OXTS not installed")
def test_eskf_meets_stage_1_4_dod_on_kitti_extract():
    sequence = load_highrate_oxts(HighRateOxtsConfig())
    config = EskfConfig()
    from scipy.stats import chi2

    lo, hi = chi2.ppf(0.025, 9), chi2.ppf(0.975, 9)
    pos_rmses, att_rmses, band_fractions = [], [], []
    for seed in range(5):
        result = run_eskf(sequence, config, seed=seed)
        pos_rmses.append(result["position_rmse"])
        att_rmses.append(result["attitude_rmse_deg"])
        nees_eval = result["nees"][50:]
        band_fractions.append(float(np.mean((nees_eval >= lo) & (nees_eval <= hi))))

    assert np.mean(pos_rmses) < 0.5, f"position RMSE {np.mean(pos_rmses):.3f} m"
    assert np.mean(att_rmses) < 1.0, f"attitude RMSE {np.mean(att_rmses):.3f} deg"
    assert np.mean(band_fractions) >= 0.90, f"NEES band {np.mean(band_fractions):.3f}"
