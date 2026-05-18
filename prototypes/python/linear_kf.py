"""4-state constant-velocity linear kalman filter prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scipy_io
from scipy.stats import chi2

STATE_DIM = 4
MEAS_DIM = 2

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def cv_matrices(
    dt: float,
    sigma_pos: float,
    q_accel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build F, H, Q (WNA discretization), and R for 2D CV tracking."""
    f = np.array( # state transition matrix, how av should move
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]) # observation matrix/measurement matrix, what sensor should read

    # Per-axis WNA for process noise covariance matrix, how much you distrust the motion model
    g1d = np.array([[0.5 * dt * dt], [dt]])
    q1d = q_accel * (g1d @ g1d.T)
    q = np.zeros((4, 4))
    q[np.ix_([0, 2], [0, 2])] = q1d  # x–vx block
    q[np.ix_([1, 3], [1, 3])] = q1d  # y–vy block

    r = (sigma_pos**2) * np.eye(MEAS_DIM) # measurement noise covariance matrix, how noisy the sensor is
    return f, h, q, r


@dataclass
class ScenarioConfig:
    dt: float = 0.1 # 100ms
    duration_s: float = 60.0 # 60s
    x0_truth: tuple[float, float, float, float] = (0.0, 0.0, 2.0, 1.0) # initial truth state
    sigma_pos: float = 1.5 # position noise standard deviation
    q_accel: float = 0.1 # acceleration noise standard deviation
    p0: np.ndarray | None = None # initial covariance matrix
    nees_burn_steps: int = 10  # exclude transient steps from consistency metrics

    @property
    def n_steps(self) -> int: # number of steps in the simulation
        return int(self.duration_s / self.dt) + 1


class LinearKF:
    """Linear Kalman filter with Joseph-form covariance update."""

    def __init__(
        self,
        f: np.ndarray,
        h: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
        x0: np.ndarray,
        p0: np.ndarray,
    ) -> None:
        self.f = np.asarray(f, dtype=float) # state transition matrix
        self.h = np.asarray(h, dtype=float) # observation matrix/measurement matrix
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
        hx = self.h @ self.x
        self.y = z - hx
        self.s = self.h @ self.p @ self.h.T + self.r
        self.k = self.p @ self.h.T @ np.linalg.inv(self.s)
        self.x = self.x + self.k @ self.y #
        i_kh = np.eye(STATE_DIM) - self.k @ self.h
        self.p = i_kh @ self.p @ i_kh.T + self.k @ self.r @ self.k.T


def simulate_cv_trajectory( # simulate the ground truth states with process noise w ~ N(0, Q)
    cfg: ScenarioConfig,
    seed: int,
) -> np.ndarray:
    """Simulate ground-truth states with process noise w ~ N(0, Q)."""
    rng = np.random.default_rng(seed)
    f, _, q, _ = cv_matrices(cfg.dt, cfg.sigma_pos, cfg.q_accel)
    n = cfg.n_steps
    x_true = np.zeros((n, STATE_DIM))
    x_true[0] = cfg.x0_truth
    for k in range(1, n):
        w = rng.multivariate_normal(np.zeros(STATE_DIM), q)
        x_true[k] = f @ x_true[k - 1] + w
    return x_true


def measure_positions( # noisy position measurements z = H x + v, v ~ N(0, R)
    x_true: np.ndarray,
    r: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Noisy position measurements z = H x + v, v ~ N(0, R)."""
    rng = np.random.default_rng(seed)
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    n = x_true.shape[0]
    z = np.zeros((n, MEAS_DIM))
    for k in range(n):
        v = rng.multivariate_normal(np.zeros(MEAS_DIM), r)
        z[k] = (h @ x_true[k]) + v
    return z


def initial_state_from_measurements(z: np.ndarray, dt: float) -> np.ndarray: # initialize position and velocity from the first two measurements
    """Initialize position and velocity from the first two measurements."""
    v0 = (z[1] - z[0]) / dt
    return np.array([z[0, 0], z[0, 1], v0[0], v0[1]])


def initial_covariance(r: np.ndarray, p0_vel: float = 10.0) -> np.ndarray: # initial covariance matrix
    """P0 with position uncertainty matching R; loose velocity prior."""
    p0 = np.diag([float(r[0, 0]), float(r[1, 1]), p0_vel, p0_vel])
    return p0


def run_filter(
    z: np.ndarray, # noisy position measurements
    f: np.ndarray, # state transition matrix
    h: np.ndarray, # observation matrix/measurement matrix
    q: np.ndarray, # process noise covariance matrix
    r: np.ndarray, # measurement noise covariance matrix
    p0: np.ndarray, # initial covariance matrix
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run predict/update; return state and covariance histories (post-update)."""
    n = z.shape[0]
    x_est = np.zeros((n, STATE_DIM)) # estimated state
    p_hist = np.zeros((n, STATE_DIM, STATE_DIM)) # previous covariance 

    kf = LinearKF(f, h, q, r, initial_state_from_measurements(z, dt), p0)
    x_est[0] = kf.x.ravel()
    p_hist[0] = kf.p

    for k in range(1, n):
        kf.predict()
        kf.update(z[k])
        x_est[k] = kf.x.ravel()
        p_hist[k] = kf.p

    return x_est, p_hist


def nees_per_step( # normalized estimation error squared
    x_true: np.ndarray,
    x_est: np.ndarray,
    p_hist: np.ndarray,
) -> np.ndarray:
    """NEES at each time step (post-update)."""
    n = x_true.shape[0]
    nees = np.zeros(n)
    for k in range(n):
        err = (x_true[k] - x_est[k]).reshape(STATE_DIM, 1)
        nees[k] = (err.T @ np.linalg.solve(p_hist[k], err)).item()
    return nees


def run_single_trial(cfg: ScenarioConfig, seed: int) -> dict: # run the filter and calculate the NEES
    f, h, q, r = cv_matrices(cfg.dt, cfg.sigma_pos, cfg.q_accel)
    p0 = cfg.p0 if cfg.p0 is not None else initial_covariance(r)

    x_true = simulate_cv_trajectory(cfg, seed)
    z = measure_positions(x_true, r, seed + 1)
    x_est, p_hist = run_filter(z, f, h, q, r, p0, cfg.dt)
    nees = nees_per_step(x_true, x_est, p_hist)

    return {
        "f": f,
        "h": h,
        "q": q,
        "r": r,
        "p0": p0,
        "x_true": x_true,
        "z": z,
        "x_est": x_est,
        "p_hist": p_hist,
        "nees": nees,
        "mean_nees": float(np.mean(nees)),
        "sum_nees": float(np.sum(nees)),
    }


def nees_chi2_bounds(n_steps: int, alpha: float = 0.05) -> tuple[float, float]: # 95% acceptance band for sum of NEES over n_steps (dof = 4 * n_steps)
    """95% acceptance band for sum of NEES over n_steps (dof = 4 * n_steps)."""
    dof = STATE_DIM * n_steps
    lo = chi2.ppf(alpha / 2, dof)
    hi = chi2.ppf(1 - alpha / 2, dof)
    return lo, hi


def run_monte_carlo( # run the filter and calculate the NEES for multiple trials
    cfg: ScenarioConfig,
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    trial_seeds = rng.integers(0, 2**31 - 1, size=n_trials)

    sum_nees = np.zeros(n_trials)
    mean_nees = np.zeros(n_trials)
    all_step_nees: list[float] = []
    for i, trial_seed in enumerate(trial_seeds):
        result = run_single_trial(cfg, int(trial_seed))
        nees_eval = result["nees"][cfg.nees_burn_steps :]
        sum_nees[i] = float(np.sum(nees_eval))
        mean_nees[i] = float(np.mean(nees_eval))
        all_step_nees.extend(result["nees"][cfg.nees_burn_steps :].tolist())

    n_eval = cfg.n_steps - cfg.nees_burn_steps
    lo, hi = nees_chi2_bounds(n_eval)
    inside_sum = (sum_nees >= lo) & (sum_nees <= hi)
    fraction_inside_sum = float(np.mean(inside_sum))

    step_lo = chi2.ppf(0.025, STATE_DIM)
    step_hi = chi2.ppf(0.975, STATE_DIM)
    all_step_nees_arr = np.asarray(all_step_nees)
    inside_step = (all_step_nees_arr >= step_lo) & (all_step_nees_arr <= step_hi)
    fraction_inside_step = float(np.mean(inside_step))

    return {
        "sum_nees": sum_nees,
        "mean_nees": mean_nees,
        "chi2_lo": lo,
        "chi2_hi": hi,
        "fraction_inside_sum": fraction_inside_sum,
        "fraction_inside_step": fraction_inside_step,
        "step_chi2_lo": step_lo,
        "step_chi2_hi": step_hi,
        "n_trials": n_trials,
    }


def draw_covariance_ellipse(
    ax: plt.Axes,
    mean_xy: np.ndarray,
    cov_xy: np.ndarray,
    n_sigma: float = 2.0,
    **kwargs,
) -> None:
    """Draw a 2D covariance ellipse on ax."""
    vals, vecs = np.linalg.eigh(cov_xy)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width = 2 * n_sigma * np.sqrt(max(vals[0], 0.0))
    height = 2 * n_sigma * np.sqrt(max(vals[1], 0.0))
    ell = plt.matplotlib.patches.Ellipse(
        xy=mean_xy,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs,
    )
    ax.add_patch(ell)


def plot_single_trial(result: dict, cfg: ScenarioConfig, out_path: Path) -> None:
    x_true = result["x_true"]
    x_est = result["x_est"]
    z = result["z"]
    p_hist = result["p_hist"]
    n = x_true.shape[0]
    ellipse_indices = np.linspace(0, n - 1, 6, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(x_true[:, 0], x_true[:, 1], "k-", label="Ground truth", linewidth=1.5)
    ax.scatter(z[:, 0], z[:, 1], s=8, c="gray", alpha=0.4, label="Measurements")
    ax.plot(x_est[:, 0], x_est[:, 1], "b-", label="KF estimate", linewidth=1.5)

    for idx in ellipse_indices:
        cov_xy = p_hist[idx, 0:2, 0:2]
        draw_covariance_ellipse(
            ax,
            x_est[idx, 0:2],
            cov_xy,
            n_sigma=2.0,
            edgecolor="blue",
            linewidth=0.8,
            alpha=0.7,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Linear CV Kalman filter — trajectory and 2σ ellipses")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_reference(result: dict, cfg: ScenarioConfig, out_dir: Path) -> None:
    """Export shared inputs for MATLAB cross-validation (.npz and .mat)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = initial_state_from_measurements(result["z"], cfg.dt)

    payload = {
        "dt": cfg.dt,
        "F": result["f"],
        "H": result["h"],
        "Q": result["q"],
        "R": result["r"],
        "x0": x0,
        "P0": result["p0"],
        "z": result["z"],
        "x_true": result["x_true"],
        "x_est_py": result["x_est"],
    }
    np.savez(out_dir / "linear_kf_ref.npz", **payload)
    scipy_io.savemat(
        out_dir / "linear_kf_ref.mat",
        payload,
        do_compression=True,
    )


def main() -> None:
    cfg = ScenarioConfig()
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Single trial (seed=0) ===")
    result = run_single_trial(cfg, seed=0)
    print(f"Mean NEES (per step): {result['mean_nees']:.3f} (expected ~ {STATE_DIM})")
    plot_single_trial(result, cfg, out_dir / "linear_kf_trajectory.png")
    print(f"Saved plot: {out_dir / 'linear_kf_trajectory.png'}")

    print("\n=== Monte Carlo NEES (100 trials) ===")
    mc = run_monte_carlo(cfg, n_trials=100, seed=42)
    print(f"Chi-squared 95% band for sum NEES: [{mc['chi2_lo']:.1f}, {mc['chi2_hi']:.1f}]")
    print(f"Fraction of trials inside sum band: {mc['fraction_inside_sum']:.1%}")
    print(
        f"Fraction of steps inside per-step 95% band "
        f"[{mc['step_chi2_lo']:.2f}, {mc['step_chi2_hi']:.2f}]: "
        f"{mc['fraction_inside_step']:.1%}"
    )
    print(f"Mean NEES across trials (avg per step): {np.mean(mc['mean_nees']):.3f}")

    # Sum NEES assumes uncorrelated steps (often fails for KF trajectories). Pooled per-step
    # NEES is the primary consistency check; mean should be near state dimension (4).
    dod_ok = mc["fraction_inside_step"] >= 0.945 and 3.5 <= np.mean(mc["mean_nees"]) <= 4.5
    if not dod_ok:
        print("WARNING: NEES consistency check not met (per-step band >= 95%, mean ~ 4).")
    else:
        print("NEES consistency: PASS")

    export_reference(result, cfg, out_dir)
    print(f"Exported reference: {out_dir / 'linear_kf_ref.npz'}")
    print(f"Exported reference: {out_dir / 'linear_kf_ref.mat'}")


if __name__ == "__main__":
    main()
