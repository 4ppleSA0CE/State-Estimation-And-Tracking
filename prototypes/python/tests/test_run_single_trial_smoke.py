from __future__ import annotations

import numpy as np
import pytest

import ekf_synthetic
import imm_synthetic
import linear_kf
import ukf_synthetic


def assert_numeric_payload_equal(left: dict, right: dict, *, skip_keys: set[str] | None = None) -> None:
    skip_keys = skip_keys or set()
    assert left.keys() == right.keys()
    for key in left:
        if key in skip_keys:
            continue
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, np.ndarray):
            np.testing.assert_allclose(left_value, right_value, rtol=0.0, atol=0.0)
        elif isinstance(left_value, list):
            np.testing.assert_allclose(np.asarray(left_value), np.asarray(right_value), rtol=0.0, atol=0.0)
        elif isinstance(left_value, float):
            assert left_value == pytest.approx(right_value, abs=0.0, rel=0.0)
        else:
            assert left_value == right_value


def test_linear_run_single_trial_is_deterministic_smoke() -> None:
    cfg = linear_kf.ScenarioConfig(dt=0.2, duration_s=1.0, nees_burn_steps=0)

    first = linear_kf.run_single_trial(cfg, seed=11)
    second = linear_kf.run_single_trial(cfg, seed=11)

    assert first["x_true"].shape == (cfg.n_steps, linear_kf.STATE_DIM)
    assert first["z"].shape == (cfg.n_steps, linear_kf.MEAS_DIM)
    assert first["x_est"].shape == (cfg.n_steps, linear_kf.STATE_DIM)
    assert first["p_hist"].shape == (cfg.n_steps, linear_kf.STATE_DIM, linear_kf.STATE_DIM)
    assert np.isfinite(first["mean_nees"])
    assert_numeric_payload_equal(first, second)


def test_ekf_run_single_trial_is_deterministic_smoke() -> None:
    cfg = ekf_synthetic.EkfScenarioConfig(dt=0.2, duration_s=1.0, nis_burn_steps=0)

    first = ekf_synthetic.run_single_trial(cfg, seed=12)
    second = ekf_synthetic.run_single_trial(cfg, seed=12)

    assert first["x_true"].shape == (cfg.n_steps, linear_kf.STATE_DIM)
    assert first["z"].shape == (cfg.n_steps, ekf_synthetic.MEAS_DIM)
    assert first["x_est"].shape == (cfg.n_steps, linear_kf.STATE_DIM)
    assert first["p_hist"].shape == (cfg.n_steps, linear_kf.STATE_DIM, linear_kf.STATE_DIM)
    assert first["s_hist"].shape == (cfg.n_steps, ekf_synthetic.MEAS_DIM, ekf_synthetic.MEAS_DIM)
    assert np.isfinite(first["mean_nis"])
    assert np.isfinite(first["pos_rmse"])
    assert_numeric_payload_equal(first, second)


def test_ukf_run_single_trial_is_deterministic_smoke() -> None:
    cfg = ekf_synthetic.EkfScenarioConfig(dt=0.2, duration_s=1.0, nis_burn_steps=0)

    first = ukf_synthetic.run_single_trial(cfg, seed=13)
    second = ukf_synthetic.run_single_trial(cfg, seed=13)

    assert first["x_true"].shape == (cfg.n_steps, linear_kf.STATE_DIM)
    assert first["z"].shape == (cfg.n_steps, ekf_synthetic.MEAS_DIM)
    assert first["x_est"].shape == (cfg.n_steps, linear_kf.STATE_DIM)
    assert first["p_hist"].shape == (cfg.n_steps, linear_kf.STATE_DIM, linear_kf.STATE_DIM)
    assert first["s_hist"].shape == (cfg.n_steps, ekf_synthetic.MEAS_DIM, ekf_synthetic.MEAS_DIM)
    assert np.isfinite(first["mean_nis"])
    assert np.isfinite(first["pos_rmse"])
    assert first["elapsed_s"] >= 0.0
    assert second["elapsed_s"] >= 0.0
    assert_numeric_payload_equal(first, second, skip_keys={"elapsed_s"})


def test_imm_run_single_trial_is_deterministic_smoke() -> None:
    cfg = imm_synthetic.ImmScenarioConfig(
        dt=0.2,
        duration_s=1.2,
        cv_duration_s=0.4,
        ct_duration_s=0.4,
    )

    first = imm_synthetic.run_single_trial(cfg, seed=14)
    second = imm_synthetic.run_single_trial(cfg, seed=14)

    assert first["x_true"].shape == (cfg.n_steps, imm_synthetic.REF_DIM)
    assert first["mode_true"].shape == (cfg.n_steps,)
    assert first["z"].shape == (cfg.n_steps, linear_kf.MEAS_DIM)
    assert first["x_imm"].shape == (cfg.n_steps, imm_synthetic.REF_DIM)
    assert first["mu_hist"].shape == (cfg.n_steps, imm_synthetic.NUM_MODES)
    np.testing.assert_allclose(first["mu_hist"].sum(axis=1), np.ones(cfg.n_steps))
    for key in ("rmse_imm", "rmse_cv", "rmse_ca", "rmse_ct"):
        assert np.isfinite(first[key])
    assert_numeric_payload_equal(first, second)
