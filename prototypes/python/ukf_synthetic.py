"""2D constant-velocity target tracked with a UKF and radar (range, bearing) measurements."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scipy_io
from scipy.stats import chi2

from ekf_synthetic import (
    MEAS_DIM,
    EkfScenarioConfig,
    cartesian_from_radar,
    cv_matrices,
    initial_covariance_radar,
    initial_state_from_radar,
    measure_radar,
    measurement_model,
    nis_per_step,
    position_rmse,
    radar_measurement_noise,
    run_single_trial as run_ekf_single_trial,
    simulate_cv_trajectory,
    wrap_angle,
)
from linear_kf import STATE_DIM, ScenarioConfig

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Julier-scaled symmetric sigma points
UKF_ALPHA = 1e-3
UKF_BETA = 2.0
UKF_KAPPA = 0.0


def unscented_weights(
    n: int,
    alpha: float = UKF_ALPHA,
    beta: float = UKF_BETA,
    kappa: float = UKF_KAPPA,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return lambda, W_m, W_c for n-dimensional state."""
    lam = alpha**2 * (n + kappa) - n
    wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
    return lam, wm, wc


def sigma_points(x: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
    """Generate 2n+1 sigma points (columns) from mean x and covariance P."""
    n = x.shape[0]
    x = np.asarray(x, dtype=float).ravel()
    p = np.asarray(p, dtype=float)
    scale = n + lam
    try:
        chol = np.linalg.cholesky(scale * p)
    except np.linalg.LinAlgError:
        p = p + 1e-9 * np.eye(n)
        chol = np.linalg.cholesky(scale * p)

    sigmas = np.zeros((n, 2 * n + 1))
    sigmas[:, 0] = x
    for i in range(n):
        sigmas[:, i + 1] = x + chol[:, i]
        sigmas[:, n + i + 1] = x - chol[:, i]
    return sigmas


def weighted_mean(
    sigmas: np.ndarray,
    wm: np.ndarray,
) -> np.ndarray:
    """Weighted mean of sigma columns."""
    return sigmas @ wm


def weighted_covariance(
    sigmas: np.ndarray,
    mean: np.ndarray,
    wc: np.ndarray,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted covariance of sigma columns; optional additive noise matrix."""
    n = sigmas.shape[0]
    diff = sigmas - mean.reshape(n, 1)
    p = diff @ np.diag(wc) @ diff.T
    if noise is not None:
        p = p + noise
    return p


def measurement_mean(z_sigmas: np.ndarray, wm: np.ndarray) -> np.ndarray:
    """Weighted mean of radar measurements; circular mean for bearing."""
    r_mean = float(np.sum(wm * z_sigmas[0, :]))
    c = float(np.sum(wm * np.cos(z_sigmas[1, :])))
    s = float(np.sum(wm * np.sin(z_sigmas[1, :])))
    return np.array([r_mean, np.arctan2(s, c)], dtype=float)


class RadarUKF:
    """Unscented Kalman filter with Joseph-form covariance update."""

    def __init__(
        self,
        f: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
        x0: np.ndarray,
        p0: np.ndarray,
        alpha: float = UKF_ALPHA,
        beta: float = UKF_BETA,
        kappa: float = UKF_KAPPA,
    ) -> None:
        self.f = np.asarray(f, dtype=float)
        self.q = np.asarray(q, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.x = np.asarray(x0, dtype=float).reshape(STATE_DIM, 1)
        self.p = np.asarray(p0, dtype=float).reshape(STATE_DIM, STATE_DIM)
        self.n = STATE_DIM
        self.lam, self.wm, self.wc = unscented_weights(self.n, alpha, beta, kappa)
        self.y: np.ndarray | None = None
        self.s: np.ndarray | None = None
        self.k: np.ndarray | None = None

    def predict(self) -> None:
        chi = sigma_points(self.x.ravel(), self.p, self.lam)
        chi_pred = self.f @ chi
        x_pred = weighted_mean(chi_pred, self.wm).reshape(STATE_DIM, 1)
        self.p = weighted_covariance(chi_pred, x_pred.ravel(), self.wc, self.q)
        self.x = x_pred

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float).reshape(MEAS_DIM, 1)
        chi = sigma_points(self.x.ravel(), self.p, self.lam)
        z_sigmas = np.zeros((MEAS_DIM, 2 * self.n + 1))
        for i in range(2 * self.n + 1):
            z_sigmas[:, i] = measurement_model(chi[:, i])

        z_pred = measurement_mean(z_sigmas, self.wm).reshape(MEAS_DIM, 1)
        y = z - z_pred
        y[1, 0] = wrap_angle(float(y[1, 0]))

        p_zz = weighted_covariance(z_sigmas, z_pred.ravel(), self.wc, self.r)
        diff_z = z_sigmas - z_pred.reshape(MEAS_DIM, 1)
        p_xz = (chi - self.x) @ np.diag(self.wc) @ diff_z.T

        self.y = y
        self.s = p_zz
        self.k = p_xz @ np.linalg.inv(p_zz)
        self.x = self.x + self.k @ y
        self.p = self.p - self.k @ p_zz @ self.k.T
        # Symmetrize and guard positive-definiteness
        self.p = 0.5 * (self.p + self.p.T)
        self.p += 1e-12 * np.eye(STATE_DIM)


def run_filter(
    z: np.ndarray,
    f: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    p0: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run predict/update; return state, covariance, innovation, and S histories."""
    n = z.shape[0]
    x_est = np.zeros((n, STATE_DIM))
    p_hist = np.zeros((n, STATE_DIM, STATE_DIM))
    y_hist = np.zeros((n, MEAS_DIM))
    s_hist = np.zeros((n, MEAS_DIM, MEAS_DIM))

    kf = RadarUKF(f, q, r, initial_state_from_radar(z, dt), p0)
    x_est[0] = kf.x.ravel()
    p_hist[0] = kf.p
    y_hist[0] = 0.0
    s_hist[0] = np.eye(MEAS_DIM)

    for k in range(1, n):
        kf.predict()
        kf.update(z[k])
        x_est[k] = kf.x.ravel()
        p_hist[k] = kf.p
        y_hist[k] = kf.y.ravel()
        s_hist[k] = kf.s

    return x_est, p_hist, y_hist, s_hist


def run_single_trial(cfg: EkfScenarioConfig, seed: int) -> dict:
    f, _, q, _ = cv_matrices(cfg.dt, sigma_pos=cfg.sigma_range, q_accel=cfg.q_accel)
    r = radar_measurement_noise(cfg.sigma_range, cfg.sigma_bearing)

    truth_cfg = ScenarioConfig(
        dt=cfg.dt,
        duration_s=cfg.duration_s,
        x0_truth=cfg.x0_truth,
        sigma_pos=cfg.sigma_range,
        q_accel=cfg.q_accel,
    )
    x_true = simulate_cv_trajectory(truth_cfg, seed)
    z = measure_radar(x_true, r, seed + 1)
    p0 = (
        cfg.p0
        if cfg.p0 is not None
        else initial_covariance_radar(z[0], cfg.sigma_range, cfg.sigma_bearing, cfg.p0_vel)
    )

    t0 = time.perf_counter()
    x_est, p_hist, y_hist, s_hist = run_filter(z, f, q, r, p0, cfg.dt)
    elapsed_s = time.perf_counter() - t0

    nis = nis_per_step(y_hist, s_hist)

    return {
        "f": f,
        "q": q,
        "r": r,
        "p0": p0,
        "x_true": x_true,
        "z": z,
        "x_est": x_est,
        "p_hist": p_hist,
        "y_hist": y_hist,
        "s_hist": s_hist,
        "nis": nis,
        "mean_nis": float(np.mean(nis[cfg.nis_burn_steps :])),
        "pos_rmse": position_rmse(x_true, x_est),
        "elapsed_s": elapsed_s,
    }


def nis_chi2_bounds(n_steps: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% acceptance band for sum of NIS over n_steps (dof = MEAS_DIM * n_steps)."""
    dof = MEAS_DIM * n_steps
    lo = chi2.ppf(alpha / 2, dof)
    hi = chi2.ppf(1 - alpha / 2, dof)
    return lo, hi


def run_monte_carlo(
    cfg: EkfScenarioConfig,
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    trial_seeds = rng.integers(0, 2**31 - 1, size=n_trials)

    sum_nis = np.zeros(n_trials)
    mean_nis = np.zeros(n_trials)
    all_step_nis: list[float] = []
    for i, trial_seed in enumerate(trial_seeds):
        result = run_single_trial(cfg, int(trial_seed))
        nis_eval = result["nis"][cfg.nis_burn_steps :]
        sum_nis[i] = float(np.sum(nis_eval))
        mean_nis[i] = float(np.mean(nis_eval))
        all_step_nis.extend(nis_eval.tolist())

    n_eval = cfg.n_steps - cfg.nis_burn_steps
    lo, hi = nis_chi2_bounds(n_eval)
    step_lo = chi2.ppf(0.025, MEAS_DIM)
    step_hi = chi2.ppf(0.975, MEAS_DIM)
    all_step_nis_arr = np.asarray(all_step_nis)

    return {
        "sum_nis": sum_nis,
        "mean_nis": mean_nis,
        "chi2_lo": lo,
        "chi2_hi": hi,
        "fraction_inside_sum": float(np.mean((sum_nis >= lo) & (sum_nis <= hi))),
        "fraction_inside_step": float(
            np.mean((all_step_nis_arr >= step_lo) & (all_step_nis_arr <= step_hi))
        ),
        "step_chi2_lo": step_lo,
        "step_chi2_hi": step_hi,
        "n_trials": n_trials,
    }


def plot_single_trial(
    result: dict,
    cfg: EkfScenarioConfig,
    out_dir: Path,
    ekf_result: dict | None = None,
) -> None:
    x_true = result["x_true"]
    x_est = result["x_est"]
    z = result["z"]
    nis = result["nis"]
    y_hist = result["y_hist"]
    t = np.arange(x_true.shape[0]) * cfg.dt

    meas_xy = np.array([cartesian_from_radar(row) for row in z])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.plot(x_true[:, 0], x_true[:, 1], "k-", label="Ground truth", linewidth=1.5)
    ax.scatter(meas_xy[:, 0], meas_xy[:, 1], s=8, c="gray", alpha=0.35, label="Meas. (Cart.)")
    ax.plot(x_est[:, 0], x_est[:, 1], "b-", label="UKF estimate", linewidth=1.5)
    if ekf_result is not None:
        x_ekf = ekf_result["x_est"]
        ax.plot(x_ekf[:, 0], x_ekf[:, 1], "r--", label="EKF estimate", linewidth=1.2, alpha=0.85)
    ax.plot(0, 0, "r^", markersize=10, label="Radar")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    pos_err = np.linalg.norm(x_true[:, 0:2] - x_est[:, 0:2], axis=1)
    ax.plot(t, pos_err, "b-", label="UKF")
    if ekf_result is not None:
        pos_err_ekf = np.linalg.norm(x_true[:, 0:2] - ekf_result["x_est"][:, 0:2], axis=1)
        ax.plot(t, pos_err_ekf, "r--", label="EKF", alpha=0.85)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t[1:], y_hist[1:, 0], label="range innov.")
    ax.plot(t[1:], y_hist[1:, 1], label="bearing innov.")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("innovation")
    ax.set_title("Innovations")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t[1:], nis[1:], "b-", label="NIS")
    ax.axhline(MEAS_DIM, color="k", linestyle="--", label=f"E[NIS]={MEAS_DIM}")
    ax.axhline(chi2.ppf(0.025, MEAS_DIM), color="gray", linestyle=":", alpha=0.7)
    ax.axhline(chi2.ppf(0.975, MEAS_DIM), color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NIS")
    ax.set_title("Normalized innovation squared")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "ukf_synthetic_summary.png", dpi=150)
    plt.close(fig)


def export_reference(result: dict, cfg: EkfScenarioConfig, out_dir: Path) -> None:
    """Export shared inputs for MATLAB cross-validation (.npz and .mat)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = initial_state_from_radar(result["z"], cfg.dt)
    payload = {
        "dt": cfg.dt,
        "F": result["f"],
        "Q": result["q"],
        "R": result["r"],
        "x0": x0,
        "P0": result["p0"],
        "z": result["z"],
        "x_true": result["x_true"],
        "x_est_py": result["x_est"],
        "ukf_alpha": UKF_ALPHA,
        "ukf_beta": UKF_BETA,
        "ukf_kappa": UKF_KAPPA,
    }
    np.savez(out_dir / "ukf_synthetic_ref.npz", **payload)
    scipy_io.savemat(out_dir / "ukf_synthetic_ref.mat", payload, do_compression=True)


def main() -> None:
    cfg = EkfScenarioConfig()
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Single trial (seed=0) ===")
    result = run_single_trial(cfg, seed=0)
    ekf_result = run_ekf_single_trial(cfg, seed=0)
    print(f"UKF mean NIS (per step): {result['mean_nis']:.3f} (expected ~ {MEAS_DIM})")
    print(f"UKF position RMSE: {result['pos_rmse']:.3f} m")
    print(f"EKF position RMSE: {ekf_result['pos_rmse']:.3f} m")
    print(f"UKF wall time: {result['elapsed_s']*1e3:.2f} ms")
    plot_single_trial(result, cfg, out_dir, ekf_result=ekf_result)
    print(f"Saved plot: {out_dir / 'ukf_synthetic_summary.png'}")

    print("\n=== Monte Carlo NIS (100 trials) ===")
    mc = run_monte_carlo(cfg, n_trials=100, seed=42)
    print(f"Chi-squared 95% band for sum NIS: [{mc['chi2_lo']:.1f}, {mc['chi2_hi']:.1f}]")
    print(f"Fraction of trials inside sum band: {mc['fraction_inside_sum']:.1%}")
    print(
        f"Fraction of steps inside per-step 95% band "
        f"[{mc['step_chi2_lo']:.2f}, {mc['step_chi2_hi']:.2f}]: "
        f"{mc['fraction_inside_step']:.1%}"
    )
    print(f"Mean NIS across trials (avg per step): {np.mean(mc['mean_nis']):.3f}")

    dod_ok = mc["fraction_inside_step"] >= 0.90 and 1.5 <= np.mean(mc["mean_nis"]) <= 2.5
    if not dod_ok:
        print("WARNING: NIS consistency check not met (per-step band >= 90%, mean ~ 2).")
    else:
        print("NIS consistency: PASS")

    export_reference(result, cfg, out_dir)
    print(f"Exported reference: {out_dir / 'ukf_synthetic_ref.mat'}")


if __name__ == "__main__":
    main()
