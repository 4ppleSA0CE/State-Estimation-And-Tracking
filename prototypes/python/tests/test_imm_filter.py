# prototypes/python/tests/test_imm_filter.py
"""IMMFilter in legacy config must reproduce imm_synthetic.run_imm; tracker config + coast behave."""
import numpy as np
from imm_synthetic import ImmScenarioConfig, run_single_trial
from linear_kf import cv_matrices, initial_covariance, initial_state_from_measurements
from imm_filter import IMMConfig, IMMFilter


def _legacy_cfg(scn: ImmScenarioConfig) -> IMMConfig:
    return IMMConfig(
        dt=scn.dt,
        sigma_pos=scn.sigma_pos,
        q_accel=scn.q_accel,
        omegas=(scn.turn_rate_rad_s,),      # single true-ω CT == imm_synthetic's third mode
        pi_diag=scn.pi_diag,
        mu0=(1 / 3, 1 / 3, 1 / 3),
        p0_vel=scn.p0_vel,
    )


def test_parity_with_imm_synthetic():
    scn = ImmScenarioConfig()
    trial = run_single_trial(scn, seed=0)
    z = trial["z"]
    x_ref, mu_ref = trial["x_imm"], trial["mu_hist"]

    _, _, _, r = cv_matrices(scn.dt, scn.sigma_pos, scn.q_accel)
    x0 = initial_state_from_measurements(z, scn.dt)
    p0 = initial_covariance(r, scn.p0_vel)

    imm = IMMFilter(_legacy_cfg(scn), r)
    imm.init_state(x0, p0)

    n = z.shape[0]
    x_got = np.zeros_like(x_ref)
    mu_got = np.zeros_like(mu_ref)
    x_got[0] = x0
    mu_got[0] = imm.mu
    for k in range(1, n):
        imm.predict()
        imm.update(z[k])
        x_got[k] = imm.state()[0]
        mu_got[k] = imm.mu

    assert np.allclose(x_got, x_ref, atol=1e-9)
    assert np.allclose(mu_got, mu_ref, atol=1e-9)


def test_coast_holds_mode_probs_and_grows_covariance():
    cfg = IMMConfig(omegas=(0.25, -0.25))          # tracker config, 4 modes
    r = np.eye(2)
    imm = IMMFilter(cfg, r)
    imm.init_state(np.array([0.0, 0.0, 5.0, 0.0]), np.diag([1.0, 1.0, 10.0, 10.0]))
    imm.predict()
    imm.update(np.array([0.5, 0.0]))
    mu_before = imm.mu.copy()
    _, p_before = imm.state()
    imm.predict()
    imm.coast()                                    # missed detection
    mu_after = imm.mu.copy()
    _, p_after = imm.state()
    assert np.allclose(mu_before, mu_after)                       # μ held
    assert np.trace(p_after) > np.trace(p_before)                # uncertainty grew


def test_predicted_measurement_matches_combined_state():
    cfg = IMMConfig(omegas=(0.25, -0.25))
    r = np.eye(2) * 2.0
    imm = IMMFilter(cfg, r)
    imm.init_state(np.array([1.0, 2.0, 3.0, 4.0]), np.diag([1.0, 1.0, 5.0, 5.0]))
    x, p = imm.state()
    z_pred, s = imm.predicted_measurement()
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    assert np.allclose(z_pred, h @ x)
    assert np.allclose(s, h @ p @ h.T + r)
