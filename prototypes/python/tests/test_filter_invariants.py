from __future__ import annotations

import numpy as np

from ekf_synthetic import measurement_jacobian, measurement_model, wrap_angle
from imm_synthetic import NUM_MODES, ImmScenarioConfig, mix_imm
from linear_kf import STATE_DIM, ScenarioConfig, cv_matrices, run_single_trial
from ukf_synthetic import sigma_points, unscented_weights, weighted_covariance, weighted_mean


def assert_symmetric_psd(matrix: np.ndarray, tol: float = 1e-9) -> None:
    matrix = np.asarray(matrix, dtype=float)
    assert matrix.shape[0] == matrix.shape[1]
    np.testing.assert_allclose(matrix, matrix.T, atol=tol, rtol=0.0)
    eigvals = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    assert np.min(eigvals) >= -tol


def test_linear_kf_matrix_shapes_and_covariance_symmetry_psd() -> None:
    cfg = ScenarioConfig(dt=0.2, duration_s=1.0, sigma_pos=1.2, q_accel=0.05)
    f, h, q, r = cv_matrices(cfg.dt, cfg.sigma_pos, cfg.q_accel)

    assert f.shape == (STATE_DIM, STATE_DIM)
    assert h.shape == (2, STATE_DIM)
    assert q.shape == (STATE_DIM, STATE_DIM)
    assert r.shape == (2, 2)
    assert_symmetric_psd(q)
    assert_symmetric_psd(r)

    result = run_single_trial(cfg, seed=7)
    assert result["x_est"].shape == (cfg.n_steps, STATE_DIM)
    assert result["p_hist"].shape == (cfg.n_steps, STATE_DIM, STATE_DIM)
    assert np.isfinite(result["x_est"]).all()
    assert np.isfinite(result["p_hist"]).all()
    for p in result["p_hist"]:
        assert_symmetric_psd(p)


def test_ekf_radar_jacobian_matches_finite_differences() -> None:
    x = np.array([120.0, -45.0, 3.0, -2.0])
    analytic = measurement_jacobian(x)
    numeric = np.zeros_like(analytic)
    eps = 1e-6

    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        forward = measurement_model(x + step)
        backward = measurement_model(x - step)
        diff = forward - backward
        diff[1] = wrap_angle(float(diff[1]))
        numeric[:, i] = diff / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_ukf_sigma_points_reconstruct_mean_and_covariance() -> None:
    mean = np.array([10.0, -4.0, 1.5, 0.25])
    cov = np.array(
        [
            [4.0, 0.4, 0.2, 0.0],
            [0.4, 3.0, 0.1, -0.2],
            [0.2, 0.1, 1.5, 0.3],
            [0.0, -0.2, 0.3, 1.0],
        ]
    )
    lam, wm, wc = unscented_weights(mean.size)
    sigmas = sigma_points(mean, cov, lam)

    reconstructed_mean = weighted_mean(sigmas, wm)
    reconstructed_cov = weighted_covariance(sigmas, reconstructed_mean, wc)

    assert sigmas.shape == (mean.size, 2 * mean.size + 1)
    np.testing.assert_allclose(reconstructed_mean, mean, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(reconstructed_cov, cov, rtol=0.0, atol=1e-9)
    assert_symmetric_psd(reconstructed_cov)


def test_imm_mixing_probabilities_and_covariances_are_well_formed() -> None:
    cfg = ImmScenarioConfig(pi_diag=0.9)
    states = [
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([1.0, -1.0, 0.5, 0.2]),
        np.array([-0.5, 0.5, 1.2, -0.1]),
    ]
    covs = [
        np.diag([1.0, 1.2, 0.5, 0.7]),
        np.diag([1.5, 1.1, 0.8, 0.6]),
        np.diag([0.9, 1.4, 0.6, 0.9]),
    ]
    mu = np.array([0.2, 0.5, 0.3])
    pi = cfg.pi_matrix

    mixed_x, mixed_p, c = mix_imm(states, covs, mu, pi)
    mu_mix = np.zeros((NUM_MODES, NUM_MODES))
    for j in range(NUM_MODES):
        for i in range(NUM_MODES):
            mu_mix[i, j] = pi[i, j] * mu[i] / c[j]

    np.testing.assert_allclose(c.sum(), 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(mu_mix.sum(axis=0), np.ones(NUM_MODES), rtol=0.0, atol=1e-12)
    assert len(mixed_x) == NUM_MODES
    assert len(mixed_p) == NUM_MODES
    for x, p in zip(mixed_x, mixed_p):
        assert x.shape == (STATE_DIM,)
        assert p.shape == (STATE_DIM, STATE_DIM)
        assert_symmetric_psd(p)
