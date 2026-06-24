import numpy as np
import pytest

from ukf_kitti import UkfConfig, sigma_points, STATE_DIM
from eskf import EskfConfig


def test_weights_are_consistent():
    cfg = UkfConfig()
    wm, wc = cfg.weights()
    assert wm.shape == (2 * STATE_DIM + 1,)
    # scaled unscented transform: only the mean weights sum to 1; the covariance
    # weights carry the (1 - alpha^2 + beta) kurtosis correction on the center point.
    np.testing.assert_allclose(wm.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(wc[0] - wm[0], 1.0 - cfg.alpha**2 + cfg.beta, atol=1e-9)
    np.testing.assert_allclose(wc[1:], wm[1:], atol=1e-12)


def test_lambda_formula():
    cfg = UkfConfig(alpha=1e-3, kappa=0.0)
    assert cfg.lam == pytest.approx(1e-3**2 * STATE_DIM - STATE_DIM)


def test_sigma_points_reconstruct_mean_and_cov():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(STATE_DIM, STATE_DIM))
    P = a @ a.T + STATE_DIM * np.eye(STATE_DIM)  # SPD
    cfg = UkfConfig()
    wm, wc = cfg.weights()
    chi = sigma_points(P, cfg.lam)
    assert chi.shape == (2 * STATE_DIM + 1, STATE_DIM)
    mean = wm @ chi
    np.testing.assert_allclose(mean, np.zeros(STATE_DIM), atol=1e-9)
    cov = np.zeros((STATE_DIM, STATE_DIM))
    for i in range(chi.shape[0]):
        cov += wc[i] * np.outer(chi[i], chi[i])
    np.testing.assert_allclose(cov, P, atol=1e-7)


def test_sigma_points_jitter_recovers_near_singular():
    P = np.zeros((STATE_DIM, STATE_DIM))
    P[0, 0] = 1.0  # rank-deficient; needs jitter
    chi = sigma_points(P, UkfConfig().lam)
    assert np.all(np.isfinite(chi))


from types import SimpleNamespace
from ukf_kitti import UnscentedKalmanFilter
from imu_mechanization import GRAVITY_ENU, NominalState
from so3 import euler_to_quat


def _nominal():
    return NominalState(
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([0.5, -0.5, 0.1]),
        q_map_imu=euler_to_quat(0.1, -0.2, 0.3),
    )


def _assert_symmetric_psd(matrix, tol=1e-8):
    np.testing.assert_allclose(matrix, matrix.T, atol=tol)
    assert np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))) >= -tol


def test_ukf_predict_keeps_covariance_symmetric_psd():
    filt = UnscentedKalmanFilter(_nominal(), UkfConfig())
    accel = np.array([0.2, -0.1, 9.9])
    gyro = np.array([0.01, -0.02, 0.03])
    for _ in range(50):
        filt.predict(accel, gyro, dt=0.01)
    _assert_symmetric_psd(filt.P)
    assert filt.P.shape == (15, 15)


def test_ukf_gps_update_keeps_covariance_psd():
    filt = UnscentedKalmanFilter(_nominal(), UkfConfig())
    filt.update_gps(filt.nominal.position + np.array([0.5, -0.3, 0.1]))
    _assert_symmetric_psd(filt.P)


def test_ukf_gps_update_moves_position_toward_measurement():
    filt = UnscentedKalmanFilter(_nominal(), UkfConfig())
    target = filt.nominal.position + np.array([3.0, -2.0, 0.0])
    start = np.linalg.norm(filt.nominal.position - target)
    filt.update_gps(target)
    assert np.linalg.norm(filt.nominal.position - target) < start


def test_ukf_stationary_converges_to_gps_truth():
    truth = np.array([10.0, -5.0, 2.0])
    nominal = NominalState(
        position=truth + np.array([3.0, -3.0, 1.0]),
        velocity=np.zeros(3),
        q_map_imu=euler_to_quat(0.0, 0.0, 0.0),
    )
    filt = UnscentedKalmanFilter(nominal, UkfConfig())
    specific_force = np.array([0.0, 0.0, -GRAVITY_ENU[2]])
    rng = np.random.default_rng(0)
    trace_start = np.trace(filt.P[0:3, 0:3])
    for k in range(2000):
        filt.predict(specific_force, np.zeros(3), dt=0.01)
        if k % 10 == 0:
            filt.update_gps(truth + rng.normal(0.0, 1.5, size=3))
    assert np.linalg.norm(filt.nominal.position - truth) < 1.0
    assert np.trace(filt.P[0:3, 0:3]) < trace_start


from ukf_kitti import run_ukf
from eskf import run_eskf


def _synthetic_sequence(n=120):
    dt = 0.01
    timestamps = np.arange(n) * dt
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


def test_run_ukf_is_deterministic():
    seq = _synthetic_sequence()
    a = run_ukf(seq, UkfConfig(), seed=7)
    b = run_ukf(seq, UkfConfig(), seed=7)
    np.testing.assert_allclose(a["x_est"], b["x_est"], rtol=0.0, atol=0.0)


def test_run_ukf_tracks_synthetic_truth():
    seq = _synthetic_sequence(n=300)
    result = run_ukf(seq, UkfConfig(), seed=1)
    assert result["x_est"].shape == (300, 3)
    assert result["position_rmse"] < 1.5
    assert np.all(np.isfinite(result["nees"]))


def test_ukf_matches_eskf_on_synthetic():
    seq = _synthetic_sequence(n=300)
    ukf = run_ukf(seq, UkfConfig(), seed=2)
    eskf = run_eskf(seq, EskfConfig(), seed=2)
    # near-linear INS problem: the two filters should agree closely
    np.testing.assert_allclose(ukf["x_est"], eskf["x_est"], atol=0.25)


def test_run_ukf_dropout_increases_error_in_window():
    seq = _synthetic_sequence(n=400)
    clean = run_ukf(seq, UkfConfig(), seed=3)
    dropped = run_ukf(seq, UkfConfig(), seed=3, dropout_window=(1.0, 3.0))
    t = clean["timestamps"]
    mask = (t >= 1.0) & (t <= 3.0)
    clean_err = np.linalg.norm(clean["x_est"][mask, 0:2] - clean["truth_positions"][mask, 0:2], axis=1)
    drop_err = np.linalg.norm(dropped["x_est"][mask, 0:2] - dropped["truth_positions"][mask, 0:2], axis=1)
    assert drop_err.mean() > clean_err.mean()


from pathlib import Path
from ukf_kitti import compare_filters, write_stage_note, plot_comparison, parse_args


def test_compare_filters_returns_both_metrics():
    seq = _synthetic_sequence(n=300)
    cmp = compare_filters(seq, UkfConfig(), seed=0, dropout_window=(1.0, 2.0))
    assert {"eskf_clean", "ukf_clean", "eskf_drop", "ukf_drop", "eskf_runtime_s", "ukf_runtime_s"} <= set(cmp)
    assert cmp["eskf_runtime_s"] > 0.0 and cmp["ukf_runtime_s"] > 0.0


def test_write_stage_note_writes_markdown(tmp_path):
    seq = _synthetic_sequence(n=200)
    cmp = compare_filters(seq, UkfConfig(), seed=0, dropout_window=(0.5, 1.0))
    out = tmp_path / "note.md"
    write_stage_note(cmp, out)
    text = out.read_text()
    assert "UKF" in text and "ESKF" in text and "rmse" in text.lower()
    assert "GPS dropout" in text  # dropout_window given -> dropout section must be written


def test_plot_comparison_writes_png(tmp_path):
    seq = _synthetic_sequence(n=200)
    cmp = compare_filters(seq, UkfConfig(), seed=0, dropout_window=(0.5, 1.0))
    out = tmp_path / "cmp.png"
    plot_comparison(cmp, out)
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_parse_args_defaults():
    args = parse_args(["--no-plot", "--dropout-start", "30", "--dropout-len", "10"])
    assert args.no_plot is True
    assert args.dropout_start == 30.0
    assert args.dropout_len == 10.0


from kitti_highrate_loader import HighRateOxtsConfig, HighRateOxtsSetupError, load_highrate_oxts, require_highrate_oxts


def _extract_available() -> bool:
    try:
        require_highrate_oxts(HighRateOxtsConfig())
        return True
    except HighRateOxtsSetupError:
        return False


@pytest.mark.skipif(not _extract_available(), reason="local KITTI Raw extract OXTS not installed")
def test_ukf_meets_stage_1_5_dod_on_kitti_extract():
    from scipy.stats import chi2

    sequence = load_highrate_oxts(HighRateOxtsConfig())
    config = UkfConfig()
    lo, hi = chi2.ppf(0.025, 9), chi2.ppf(0.975, 9)
    pos, att, band = [], [], []
    for seed in range(5):
        r = run_ukf(sequence, config, seed=seed)
        pos.append(r["position_rmse"])
        att.append(r["attitude_rmse_deg"])
        nv = r["nees"][50:]
        band.append(float(np.mean((nv >= lo) & (nv <= hi))))
    assert np.mean(pos) < 0.5, f"position RMSE {np.mean(pos):.3f} m"
    assert np.mean(att) < 1.0, f"attitude RMSE {np.mean(att):.3f} deg"
    assert np.mean(band) >= 0.90, f"NEES band {np.mean(band):.3f}"
