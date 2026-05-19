"""IMM tracker (CV-KF, CA-KF, CT-UKF) on a maneuvering 2D target with position measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scipy_io
from linear_kf import (
    MEAS_DIM,
    STATE_DIM,
    LinearKF,
    cv_matrices,
    initial_covariance,
    initial_state_from_measurements,
    measure_positions,
)
from ukf_synthetic import (
    sigma_points,
    unscented_weights,
    weighted_covariance,
    weighted_mean,
)

REF_DIM = STATE_DIM  # [x, y, vx, vy] for mixing and combined output
CA_DIM = 6
NUM_MODES = 3
MODE_CV = 0
MODE_CA = 1
MODE_CT = 2
MODE_NAMES = ("CV", "CA", "CT")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def ca_matrices(
    dt: float,
    sigma_pos: float,
    q_accel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """F, H, Q, R for 6-state constant-acceleration model (x, vx, ax per axis)."""
    f1 = np.array(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )
    f = np.zeros((CA_DIM, CA_DIM))
    f[np.ix_([0, 2, 4], [0, 2, 4])] = f1
    f[np.ix_([1, 3, 5], [1, 3, 5])] = f1

    h = np.zeros((MEAS_DIM, CA_DIM))
    h[0, 0] = 1.0
    h[1, 1] = 1.0

    g = np.array([[0.5 * dt * dt], [dt], [1.0]])
    q1 = q_accel * (g @ g.T)
    q = np.zeros((CA_DIM, CA_DIM))
    q[np.ix_([0, 2, 4], [0, 2, 4])] = q1
    q[np.ix_([1, 3, 5], [1, 3, 5])] = q1

    r = (sigma_pos**2) * np.eye(MEAS_DIM)
    return f, h, q, r


def ct_propagate_state(x: np.ndarray, dt: float, omega: float) -> np.ndarray:
    """Coordinated turn on [x, y, vx, vy] with constant turn rate omega (rad/s)."""
    x = np.asarray(x, dtype=float).ravel()
    px, py, vx, vy = x
    if abs(omega) < 1e-8:
        return np.array([px + vx * dt, py + vy * dt, vx, vy], dtype=float)
    s = np.sin(omega * dt)
    c = np.cos(omega * dt)
    vx_n = c * vx - s * vy
    vy_n = s * vx + c * vy
    px_n = px + (vx * s + vy * (1.0 - c)) / omega
    py_n = py + (vy * s - vx * (1.0 - c)) / omega
    return np.array([px_n, py_n, vx_n, vy_n], dtype=float)


def ref_from_cv(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(x, dtype=float).ravel()[:REF_DIM], np.asarray(p, dtype=float)[:REF_DIM, :REF_DIM]


def ref_from_ca(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CA state order: [x, y, vx, vy, ax, ay]."""
    x = np.asarray(x, dtype=float).ravel()
    p = np.asarray(p, dtype=float)
    return x[:4].copy(), p[:4, :4].copy()


def ref_to_ca(x_ref: np.ndarray, p_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_ref = np.asarray(x_ref, dtype=float).ravel()
    p_ref = np.asarray(p_ref, dtype=float)
    x = np.zeros(CA_DIM)
    x[0], x[1], x[2], x[3] = x_ref[0], x_ref[1], x_ref[2], x_ref[3]
    p = np.diag([1.0, 1.0, 1.0, 1.0, 0.5, 0.5]) * 10.0
    p[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])] = p_ref
    return x, p


def mix_imm(
    states: list[np.ndarray],
    covs: list[np.ndarray],
    mu: np.ndarray,
    pi: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """IMM mixing step; states/covs in reference [x,y,vx,vy] coordinates."""
    c = pi.T @ mu
    c = np.maximum(c, 1e-12)
    mu_mix = np.zeros((NUM_MODES, NUM_MODES))
    for j in range(NUM_MODES):
        for i in range(NUM_MODES):
            mu_mix[i, j] = pi[i, j] * mu[i] / c[j]

    mixed_x: list[np.ndarray] = []
    mixed_p: list[np.ndarray] = []
    for j in range(NUM_MODES):
        x0 = np.zeros(REF_DIM)
        for i in range(NUM_MODES):
            x0 += mu_mix[i, j] * states[i]
        p0 = np.zeros((REF_DIM, REF_DIM))
        for i in range(NUM_MODES):
            dx = (states[i] - x0).reshape(REF_DIM, 1)
            p0 += mu_mix[i, j] * (covs[i] + dx @ dx.T)
        mixed_x.append(x0)
        mixed_p.append(p0)
    return mixed_x, mixed_p, c


def gaussian_likelihood(y: np.ndarray, s: np.ndarray) -> float:
    """Scalar measurement likelihood N(y; 0, S)."""
    y = np.asarray(y, dtype=float).ravel()
    s = np.asarray(s, dtype=float)
    m = y.shape[0]
    sign, logdet = np.linalg.slogdet(s)
    if sign <= 0:
        return 1e-300
    quad = float(y.T @ np.linalg.solve(s, y))
    return float(np.exp(-0.5 * (m * np.log(2 * np.pi) + logdet + quad)))


class CvModelFilter:
    """4-state CV linear Kalman filter."""

    def __init__(
        self,
        dt: float,
        sigma_pos: float,
        q_accel: float,
        r: np.ndarray,
    ) -> None:
        self.f, self.h, self.q, self.r = cv_matrices(dt, sigma_pos, q_accel)
        self.kf: LinearKF | None = None

    def set_state(self, x_ref: np.ndarray, p_ref: np.ndarray) -> None:
        x0 = np.asarray(x_ref, dtype=float).reshape(REF_DIM, 1)
        p0 = np.asarray(p_ref, dtype=float)
        self.kf = LinearKF(self.f, self.h, self.q, self.r, x0, p0)

    def predict(self) -> None:
        assert self.kf is not None
        self.kf.predict()

    def update(self, z: np.ndarray) -> float:
        assert self.kf is not None
        self.kf.update(z)
        assert self.kf.y is not None and self.kf.s is not None
        return gaussian_likelihood(self.kf.y, self.kf.s)

    def ref_state(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.kf is not None
        return ref_from_cv(self.kf.x, self.kf.p)


class CaModelFilter:
    """6-state CA linear Kalman filter."""

    def __init__(
        self,
        dt: float,
        sigma_pos: float,
        q_accel: float,
        r: np.ndarray,
    ) -> None:
        self.f, self.h, self.q, self.r = ca_matrices(dt, sigma_pos, q_accel)
        self.x = np.zeros((CA_DIM, 1))
        self.p = np.eye(CA_DIM)
        self.y: np.ndarray | None = None
        self.s: np.ndarray | None = None

    def set_state(self, x_ref: np.ndarray, p_ref: np.ndarray) -> None:
        x0, p0 = ref_to_ca(x_ref, p_ref)
        self.x = x0.reshape(CA_DIM, 1)
        self.p = p0

    def predict(self) -> None:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q

    def update(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float).reshape(MEAS_DIM, 1)
        hx = self.h @ self.x
        y = z - hx
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        i_kh = np.eye(CA_DIM) - k @ self.h
        self.p = i_kh @ self.p @ i_kh.T + k @ self.r @ k.T
        self.y = y
        self.s = s
        return gaussian_likelihood(y, s)

    def ref_state(self) -> tuple[np.ndarray, np.ndarray]:
        return ref_from_ca(self.x, self.p)


class CtModelFilter:
    """4-state coordinated-turn UKF (fixed turn rate in dynamics)."""

    def __init__(self, dt: float, omega: float, q_accel: float, r: np.ndarray) -> None:
        self.dt = dt
        self.omega = omega
        self.r = r
        g1d = np.array([[0.5 * dt * dt], [dt]])
        q1d = q_accel * (g1d @ g1d.T)
        self.q = np.zeros((REF_DIM, REF_DIM))
        self.q[np.ix_([0, 2], [0, 2])] = q1d
        self.q[np.ix_([1, 3], [1, 3])] = q1d
        self.lam, self.wm, self.wc = unscented_weights(REF_DIM)
        self.x = np.zeros((REF_DIM, 1))
        self.p = np.eye(REF_DIM)
        self.y: np.ndarray | None = None
        self.s: np.ndarray | None = None

    def set_state(self, x_ref: np.ndarray, p_ref: np.ndarray) -> None:
        self.x = np.asarray(x_ref, dtype=float).reshape(REF_DIM, 1)
        self.p = np.asarray(p_ref, dtype=float)

    def _f(self, chi: np.ndarray) -> np.ndarray:
        n_sig = chi.shape[1]
        out = np.zeros_like(chi)
        for i in range(n_sig):
            out[:, i] = ct_propagate_state(chi[:, i], self.dt, self.omega)
        return out

    def predict(self) -> None:
        chi = sigma_points(self.x.ravel(), self.p, self.lam)
        chi_pred = self._f(chi)
        x_pred = weighted_mean(chi_pred, self.wm).reshape(REF_DIM, 1)
        self.p = weighted_covariance(chi_pred, x_pred.ravel(), self.wc, self.q)
        self.x = x_pred

    def update(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float).reshape(MEAS_DIM, 1)
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        chi = sigma_points(self.x.ravel(), self.p, self.lam)
        z_sigmas = h @ chi
        z_pred = weighted_mean(z_sigmas, self.wm).reshape(MEAS_DIM, 1)
        y = z - z_pred
        p_zz = weighted_covariance(z_sigmas, z_pred.ravel(), self.wc, self.r)
        diff_z = z_sigmas - z_pred.reshape(MEAS_DIM, 1)
        p_xz = (chi - self.x) @ np.diag(self.wc) @ diff_z.T
        k = p_xz @ np.linalg.inv(p_zz)
        self.x = self.x + k @ y
        self.p = self.p - k @ p_zz @ k.T
        self.p = 0.5 * (self.p + self.p.T) + 1e-12 * np.eye(REF_DIM)
        self.y = y
        self.s = p_zz
        return gaussian_likelihood(y, p_zz)

    def ref_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.ravel(), self.p


@dataclass
class ImmScenarioConfig:
    dt: float = 0.1
    duration_s: float = 30.0
    cv_duration_s: float = 10.0
    ct_duration_s: float = 10.0
    x0_truth: tuple[float, float, float, float] = (0.0, 0.0, 8.0, 0.0)
    turn_rate_deg_s: float = 30.0
    sigma_pos: float = 1.0
    q_accel: float = 0.05
    p0_vel: float = 10.0
    pi_diag: float = 0.97
    mu0: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    @property
    def n_steps(self) -> int:
        return int(self.duration_s / self.dt) + 1

    @property
    def turn_rate_rad_s(self) -> float:
        return np.deg2rad(self.turn_rate_deg_s)

    @property
    def pi_matrix(self) -> np.ndarray:
        off = (1.0 - self.pi_diag) / (NUM_MODES - 1)
        pi = np.full((NUM_MODES, NUM_MODES), off)
        np.fill_diagonal(pi, self.pi_diag)
        return pi


def simulate_maneuvering_trajectory(cfg: ImmScenarioConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Truth: CV → CT (constant turn) → CV; returns x_true and mode index per step."""
    rng = np.random.default_rng(seed)
    f_cv, _, q_cv, _ = cv_matrices(cfg.dt, cfg.sigma_pos, cfg.q_accel)
    n = cfg.n_steps
    n_cv1 = int(cfg.cv_duration_s / cfg.dt)
    n_ct = int(cfg.ct_duration_s / cfg.dt)
    x_true = np.zeros((n, REF_DIM))
    mode_true = np.zeros(n, dtype=int)
    x_true[0] = cfg.x0_truth
    mode_true[0] = MODE_CV

    for k in range(1, n):
        if k <= n_cv1:
            w = rng.multivariate_normal(np.zeros(REF_DIM), q_cv)
            x_true[k] = f_cv @ x_true[k - 1] + w
            mode_true[k] = MODE_CV
        elif k <= n_cv1 + n_ct:
            x_true[k] = ct_propagate_state(x_true[k - 1], cfg.dt, cfg.turn_rate_rad_s)
            mode_true[k] = MODE_CT
        else:
            w = rng.multivariate_normal(np.zeros(REF_DIM), q_cv)
            x_true[k] = f_cv @ x_true[k - 1] + w
            mode_true[k] = MODE_CV

    return x_true, mode_true


def combined_estimate(
    states: list[np.ndarray],
    covs: list[np.ndarray],
    mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(REF_DIM)
    for j in range(NUM_MODES):
        x += mu[j] * states[j]
    p = np.zeros((REF_DIM, REF_DIM))
    for j in range(NUM_MODES):
        dx = (states[j] - x).reshape(REF_DIM, 1)
        p += mu[j] * (covs[j] + dx @ dx.T)
    return x, p


def run_imm(
    z: np.ndarray,
    cfg: ImmScenarioConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run IMM; return combined state history, mode probabilities, per-model ref states."""
    n = z.shape[0]
    _, _, _, r = cv_matrices(cfg.dt, cfg.sigma_pos, cfg.q_accel)
    p0 = initial_covariance(r, cfg.p0_vel)
    x0 = initial_state_from_measurements(z, cfg.dt)

    filters: list[CvModelFilter | CaModelFilter | CtModelFilter] = [
        CvModelFilter(cfg.dt, cfg.sigma_pos, cfg.q_accel, r),
        CaModelFilter(cfg.dt, cfg.sigma_pos, cfg.q_accel, r),
        CtModelFilter(cfg.dt, cfg.turn_rate_rad_s, cfg.q_accel, r),
    ]
    for f in filters:
        f.set_state(x0, p0)

    mu = np.array(cfg.mu0, dtype=float)
    pi = cfg.pi_matrix

    x_comb = np.zeros((n, REF_DIM))
    mu_hist = np.zeros((n, NUM_MODES))
    x_comb[0] = x0
    mu_hist[0] = mu

    for k in range(1, n):
        states = [f.ref_state()[0] for f in filters]
        covs = [f.ref_state()[1] for f in filters]
        mixed_x, mixed_p, c = mix_imm(states, covs, mu, pi)

        likelihoods = np.zeros(NUM_MODES)
        for j, filt in enumerate(filters):
            filt.set_state(mixed_x[j], mixed_p[j])
            filt.predict()
            likelihoods[j] = filt.update(z[k])

        states = [f.ref_state()[0] for f in filters]
        covs = [f.ref_state()[1] for f in filters]
        denom = float(np.sum(likelihoods * c))
        if denom < 1e-300:
            mu = np.array(cfg.mu0, dtype=float)
        else:
            mu = (likelihoods * c) / denom
            mu = np.maximum(mu, 1e-12)
            mu /= mu.sum()

        x_comb[k], _ = combined_estimate(states, covs, mu)
        mu_hist[k] = mu

    return x_comb, mu_hist, mu


def run_single_model(
    z: np.ndarray,
    cfg: ImmScenarioConfig,
    mode: int,
) -> np.ndarray:
    """Run one model only (no IMM) for baseline comparison."""
    n = z.shape[0]
    _, _, _, r = cv_matrices(cfg.dt, cfg.sigma_pos, cfg.q_accel)
    p0 = initial_covariance(r, cfg.p0_vel)
    x0 = initial_state_from_measurements(z, cfg.dt)

    if mode == MODE_CV:
        filt: CvModelFilter | CaModelFilter | CtModelFilter = CvModelFilter(
            cfg.dt, cfg.sigma_pos, cfg.q_accel, r
        )
    elif mode == MODE_CA:
        filt = CaModelFilter(cfg.dt, cfg.sigma_pos, cfg.q_accel, r)
    else:
        filt = CtModelFilter(cfg.dt, cfg.turn_rate_rad_s, cfg.q_accel, r)

    filt.set_state(x0, p0)
    x_est = np.zeros((n, REF_DIM))
    x_est[0] = x0
    for k in range(1, n):
        filt.predict()
        filt.update(z[k])
        x_est[k] = filt.ref_state()[0]
    return x_est


def position_rmse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    err = x_true[:, 0:2] - x_est[:, 0:2]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def mode_switch_latency(
    mu_hist: np.ndarray,
    mode_true: np.ndarray,
    cfg: ImmScenarioConfig,
    threshold: float = 0.5,
) -> list[float]:
    """Seconds after true regime change until dominant mu matches true mode."""
    n_cv1 = int(cfg.cv_duration_s / cfg.dt)
    n_ct = int(cfg.ct_duration_s / cfg.dt)
    switches = [
        (n_cv1, MODE_CT),
        (n_cv1 + n_ct, MODE_CV),
    ]
    latencies: list[float] = []
    for step, true_mode in switches:
        found = False
        for k in range(step, cfg.n_steps):
            if mu_hist[k, true_mode] >= threshold:
                latencies.append((k - step) * cfg.dt)
                found = True
                break
        if not found:
            latencies.append(float("inf"))
    return latencies


def run_single_trial(cfg: ImmScenarioConfig, seed: int) -> dict:
    _, _, _, r = cv_matrices(cfg.dt, cfg.sigma_pos, cfg.q_accel)
    x_true, mode_true = simulate_maneuvering_trajectory(cfg, seed)
    z = measure_positions(x_true, r, seed + 1)

    x_imm, mu_hist, mu_final = run_imm(z, cfg)
    x_cv = run_single_model(z, cfg, MODE_CV)
    x_ca = run_single_model(z, cfg, MODE_CA)
    x_ct = run_single_model(z, cfg, MODE_CT)

    rmse_imm = position_rmse(x_true, x_imm)
    rmse_cv = position_rmse(x_true, x_cv)
    rmse_ca = position_rmse(x_true, x_ca)
    rmse_ct = position_rmse(x_true, x_ct)
    latencies = mode_switch_latency(mu_hist, mode_true, cfg)

    return {
        "r": r,
        "x_true": x_true,
        "mode_true": mode_true,
        "z": z,
        "x_imm": x_imm,
        "mu_hist": mu_hist,
        "mu_final": mu_final,
        "x_cv": x_cv,
        "x_ca": x_ca,
        "x_ct": x_ct,
        "rmse_imm": rmse_imm,
        "rmse_cv": rmse_cv,
        "rmse_ca": rmse_ca,
        "rmse_ct": rmse_ct,
        "switch_latencies_s": latencies,
        "pi": cfg.pi_matrix,
    }


def plot_single_trial(result: dict, cfg: ImmScenarioConfig, out_dir: Path) -> None:
    x_true = result["x_true"]
    x_imm = result["x_imm"]
    mu_hist = result["mu_hist"]
    z = result["z"]
    t = np.arange(x_true.shape[0]) * cfg.dt
    t_cv1 = cfg.cv_duration_s
    t_ct_end = cfg.cv_duration_s + cfg.ct_duration_s

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.plot(x_true[:, 0], x_true[:, 1], "k-", label="Ground truth", linewidth=1.5)
    ax.scatter(z[:, 0], z[:, 1], s=6, c="gray", alpha=0.3, label="Measurements")
    ax.plot(x_imm[:, 0], x_imm[:, 1], "b-", label="IMM combined", linewidth=1.5)
    ax.plot(result["x_cv"][:, 0], result["x_cv"][:, 1], "r--", alpha=0.6, label="CV only")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for m, name, color in zip(range(NUM_MODES), MODE_NAMES, ("C0", "C1", "C2")):
        ax.plot(t, mu_hist[:, m], color=color, label=name)
    ax.axvline(t_cv1, color="gray", linestyle=":", label="CV→CT")
    ax.axvline(t_ct_end, color="gray", linestyle="--", label="CT→CV")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mode probability")
    ax.set_title("IMM mode probabilities")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    err_imm = np.linalg.norm(x_true[:, 0:2] - x_imm[:, 0:2], axis=1)
    err_cv = np.linalg.norm(x_true[:, 0:2] - result["x_cv"][:, 0:2], axis=1)
    ax.plot(t, err_imm, "b-", label="IMM")
    ax.plot(t, err_cv, "r--", label="CV only", alpha=0.8)
    ax.axvline(t_cv1, color="gray", linestyle=":")
    ax.axvline(t_ct_end, color="gray", linestyle="--")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    labels = ["IMM", "CV", "CA", "CT"]
    rmses = [result["rmse_imm"], result["rmse_cv"], result["rmse_ca"], result["rmse_ct"]]
    colors = ["b", "r", "orange", "green"]
    ax.bar(labels, rmses, color=colors, alpha=0.8)
    ax.set_ylabel("RMSE [m]")
    ax.set_title("Position RMSE")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "imm_synthetic_summary.png", dpi=150)
    plt.close(fig)


def export_reference(result: dict, cfg: ImmScenarioConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dt": cfg.dt,
        "R": result["r"],
        "Pi": result["pi"],
        "z": result["z"],
        "x_true": result["x_true"],
        "mode_true": result["mode_true"],
        "x_imm_py": result["x_imm"],
        "mu_hist_py": result["mu_hist"],
        "turn_rate_rad_s": cfg.turn_rate_rad_s,
        "cv_duration_s": cfg.cv_duration_s,
        "ct_duration_s": cfg.ct_duration_s,
    }
    np.savez(out_dir / "imm_synthetic_ref.npz", **payload)
    scipy_io.savemat(out_dir / "imm_synthetic_ref.mat", payload, do_compression=True)


def main() -> None:
    cfg = ImmScenarioConfig()
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Single trial (seed=0) ===")
    result = run_single_trial(cfg, seed=0)
    print(f"IMM position RMSE: {result['rmse_imm']:.3f} m")
    print(f"CV-only RMSE:      {result['rmse_cv']:.3f} m")
    print(f"CA-only RMSE:      {result['rmse_ca']:.3f} m")
    print(f"CT-only RMSE:      {result['rmse_ct']:.3f} m")
    print(f"Switch latencies (s): {result['switch_latencies_s']}")
    print(f"Final mode probs: {result['mu_final']}")

    plot_single_trial(result, cfg, out_dir)
    print(f"Saved plot: {out_dir / 'imm_synthetic_summary.png'}")

    best_single = min(result["rmse_cv"], result["rmse_ca"], result["rmse_ct"])
    dod_rmse = result["rmse_imm"] < best_single
    dod_switch = all(lat <= 2.0 for lat in result["switch_latencies_s"])
    if dod_rmse:
        print("DOD RMSE: PASS (IMM beats best single model)")
    else:
        print("DOD RMSE: WARNING (IMM did not beat best single model)")
    if dod_switch:
        print("DOD mode switch: PASS (<= 2 s)")
    else:
        print("DOD mode switch: WARNING")

    export_reference(result, cfg, out_dir)
    print(f"Exported reference: {out_dir / 'imm_synthetic_ref.mat'}")


if __name__ == "__main__":
    main()
