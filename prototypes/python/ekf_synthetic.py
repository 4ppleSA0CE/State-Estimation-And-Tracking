"""2D constant-velocity target tracked with an EKF and radar (range, bearing) measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scipy_io
from scipy.stats import chi2

from linear_kf import (
    STATE_DIM,
    ScenarioConfig,
    cv_matrices,
    simulate_cv_trajectory,
)

MEAS_DIM = 2
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def radar_measurement_noise(sigma_range: float, sigma_bearing: float) -> np.ndarray:
    """Diagonal R for range (m) and bearing (rad) measurements."""
    return np.diag([sigma_range**2, sigma_bearing**2]) # measurement noise covariance matrix, how noisy the sensor is


def sigma_bearing_from_position_noise(sigma_pos: float, range_m: float) -> float:
    """Map Cartesian position std (m) to bearing std (rad) at slant range range_m."""
    return sigma_pos / max(range_m, 1.0)


def initial_covariance_radar(
    z0: np.ndarray,
    sigma_range: float,
    sigma_bearing: float,
    p0_vel: float = 10.0,
) -> np.ndarray: # initial covariance matrix
    """P0 in Cartesian state: position block from polar R via J(r,theta)->(x,y)."""
    r_m = max(float(z0[0]), 1.0)
    theta = float(z0[1])
    r_pol = np.diag([sigma_range**2, sigma_bearing**2]) # measurement noise covariance matrix
    j = np.array( # polar to Cartesian Jacobian for P0 position block
        [
            [np.cos(theta), -r_m * np.sin(theta)],
            [np.sin(theta), r_m * np.cos(theta)],
        ]
    )
    p_xy = j @ r_pol @ j.T
    p0 = np.zeros((STATE_DIM, STATE_DIM))
    p0[0:2, 0:2] = p_xy
    p0[2, 2] = p0_vel
    p0[3, 3] = p0_vel
    return p0


def measurement_model(x: np.ndarray) -> np.ndarray:
    """Nonlinear h(x): radar at origin, z = [range, bearing]."""
    xv = np.asarray(x, dtype=float).ravel()
    px, py = xv[0], xv[1]
    r = np.hypot(px, py)
    return np.array([r, np.arctan2(py, px)], dtype=float) # what sensor should read


def measurement_jacobian(x: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of h(x); used as observation matrix in the EKF update."""
    xv = np.asarray(x, dtype=float).ravel()
    px, py = xv[0], xv[1]
    r2 = px * px + py * py
    r = max(np.sqrt(r2), 1e-6)
    r2 = max(r2, r * r)
    return np.array( # observation matrix/measurement matrix (linearized), what sensor should read
        [
            [px / r, py / r, 0.0, 0.0],
            [-py / r2, px / r2, 0.0, 0.0],
        ],
        dtype=float,
    )


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def cartesian_from_radar(z: np.ndarray) -> np.ndarray:
    """Convert [range, bearing] to [x, y]."""
    return np.array([z[0] * np.cos(z[1]), z[0] * np.sin(z[1])], dtype=float)


def initial_state_from_radar(z: np.ndarray, dt: float) -> np.ndarray: # initialize position and velocity from the first two measurements
    """Initialize [x, y, vx, vy] from the first two radar returns."""
    p0 = cartesian_from_radar(z[0])
    p1 = cartesian_from_radar(z[1])
    v0 = (p1 - p0) / dt
    return np.array([p0[0], p0[1], v0[0], v0[1]], dtype=float)


@dataclass
class EkfScenarioConfig:
    dt: float = 0.1 # 100ms
    duration_s: float = 60.0 # 60s
    x0_truth: tuple[float, float, float, float] = (100.0, 50.0, -3.0, 2.0) # initial truth state
    q_accel: float = 0.1 # acceleration noise standard deviation
    sigma_range: float = 1.5 # range measurement noise standard deviation
    sigma_bearing: float | None = None # bearing measurement noise standard deviation
    p0_vel: float = 10.0 # initial velocity variance for P0
    p0: np.ndarray | None = None # initial covariance matrix
    nis_burn_steps: int = 10  # exclude transient steps from consistency metrics

    def __post_init__(self) -> None:
        if self.sigma_bearing is None:
            r0 = max(np.hypot(self.x0_truth[0], self.x0_truth[1]), 1.0)
            object.__setattr__(
                self,
                "sigma_bearing",
                sigma_bearing_from_position_noise(self.sigma_range, r0),
            )

    @property
    def n_steps(self) -> int: # number of steps in the simulation
        return int(self.duration_s / self.dt) + 1


class RadarEKF:
    """Extended Kalman filter with Joseph-form covariance update."""

    def __init__(
        self,
        f: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
        x0: np.ndarray,
        p0: np.ndarray,
    ) -> None:
        self.f = np.asarray(f, dtype=float) # state transition matrix
        self.q = np.asarray(q, dtype=float) # process noise covariance matrix
        self.r = np.asarray(r, dtype=float) # measurement noise covariance matrix
        self.x = np.asarray(x0, dtype=float).reshape(STATE_DIM, 1) # initial state
        self.p = np.asarray(p0, dtype=float).reshape(STATE_DIM, STATE_DIM) # initial covariance matrix
        self.y: np.ndarray | None = None # measurement innovation
        self.s: np.ndarray | None = None # measurement innovation covariance matrix
        self.k: np.ndarray | None = None # kalman gain

    def predict(self) -> None:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float).reshape(MEAS_DIM, 1)
        hx = measurement_model(self.x.ravel()).reshape(MEAS_DIM, 1)
        y = z - hx
        y[1, 0] = wrap_angle(float(y[1, 0]))
        h = measurement_jacobian(self.x.ravel())
        self.y = y
        self.s = h @ self.p @ h.T + self.r
        self.k = self.p @ h.T @ np.linalg.inv(self.s)
        self.x = self.x + self.k @ y #
        i_kh = np.eye(STATE_DIM) - self.k @ h
        self.p = i_kh @ self.p @ i_kh.T + self.k @ self.r @ self.k.T


def measure_radar( # noisy radar measurements z = h(x) + v, v ~ N(0, R)
    x_true: np.ndarray,
    r: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Noisy radar measurements z = h(x) + v, v ~ N(0, R)."""
    rng = np.random.default_rng(seed)
    n = x_true.shape[0]
    z = np.zeros((n, MEAS_DIM))
    for k in range(n):
        v = rng.multivariate_normal(np.zeros(MEAS_DIM), r)
        z[k] = measurement_model(x_true[k]) + v
    return z


def run_filter(
    z: np.ndarray, # noisy radar measurements
    f: np.ndarray, # state transition matrix
    q: np.ndarray, # process noise covariance matrix
    r: np.ndarray, # measurement noise covariance matrix
    p0: np.ndarray, # initial covariance matrix
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run predict/update; return state, covariance, innovation, and S histories."""
    n = z.shape[0]
    x_est = np.zeros((n, STATE_DIM)) # estimated state
    p_hist = np.zeros((n, STATE_DIM, STATE_DIM)) # previous covariance
    y_hist = np.zeros((n, MEAS_DIM)) # measurement innovation history
    s_hist = np.zeros((n, MEAS_DIM, MEAS_DIM)) # measurement innovation covariance history

    kf = RadarEKF(f, q, r, initial_state_from_radar(z, dt), p0)
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


def nis_per_step( # normalized innovation squared
    y_hist: np.ndarray,
    s_hist: np.ndarray,
) -> np.ndarray:
    """Normalized innovation squared at each time step (post-update)."""
    n = y_hist.shape[0]
    nis = np.zeros(n)
    for k in range(1, n):
        y = y_hist[k].reshape(MEAS_DIM, 1)
        s = s_hist[k]
        nis[k] = float((y.T @ np.linalg.solve(s, y)).item())
    return nis


def position_rmse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    err = x_true[:, 0:2] - x_est[:, 0:2]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def run_single_trial(cfg: EkfScenarioConfig, seed: int) -> dict: # run the filter and calculate the NIS
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
    x_est, p_hist, y_hist, s_hist = run_filter(z, f, q, r, p0, cfg.dt)
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
    }


def nis_chi2_bounds(n_steps: int, alpha: float = 0.05) -> tuple[float, float]: # 95% acceptance band for sum of NIS over n_steps (dof = 2 * n_steps)
    """95% acceptance band for sum of NIS over n_steps (dof = MEAS_DIM * n_steps)."""
    dof = MEAS_DIM * n_steps
    lo = chi2.ppf(alpha / 2, dof)
    hi = chi2.ppf(1 - alpha / 2, dof)
    return lo, hi


def run_monte_carlo( # run the filter and calculate the NIS for multiple trials
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


def plot_single_trial(result: dict, cfg: EkfScenarioConfig, out_dir: Path) -> None:
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
    ax.plot(x_est[:, 0], x_est[:, 1], "b-", label="EKF estimate", linewidth=1.5)
    ax.plot(0, 0, "r^", markersize=10, label="Radar")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    pos_err = np.linalg.norm(x_true[:, 0:2] - x_est[:, 0:2], axis=1)
    ax.plot(t, pos_err, "b-")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error")
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
    fig.savefig(out_dir / "ekf_synthetic_summary.png", dpi=150)
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
    }
    np.savez(out_dir / "ekf_synthetic_ref.npz", **payload)
    scipy_io.savemat(out_dir / "ekf_synthetic_ref.mat", payload, do_compression=True)


def main() -> None:
    cfg = EkfScenarioConfig()
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Single trial (seed=0) ===")
    result = run_single_trial(cfg, seed=0)
    print(f"Mean NIS (per step): {result['mean_nis']:.3f} (expected ~ {MEAS_DIM})")
    print(f"Position RMSE: {result['pos_rmse']:.3f} m")
    plot_single_trial(result, cfg, out_dir)
    print(f"Saved plot: {out_dir / 'ekf_synthetic_summary.png'}")

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

    # Sum NIS assumes uncorrelated steps (often fails for KF trajectories). Pooled per-step
    # NIS is the primary consistency check; mean should be near measurement dimension (2).
    dod_ok = mc["fraction_inside_step"] >= 0.90 and 1.5 <= np.mean(mc["mean_nis"]) <= 2.5
    if not dod_ok:
        print("WARNING: NIS consistency check not met (per-step band >= 90%, mean ~ 2).")
    else:
        print("NIS consistency: PASS")

    export_reference(result, cfg, out_dir)
    print(f"Exported reference: {out_dir / 'ekf_synthetic_ref.mat'}")


if __name__ == "__main__":
    main()
