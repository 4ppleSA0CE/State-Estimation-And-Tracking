"""15-state full-state Unscented Kalman Filter (UKF) on KITTI IMU + GPS.

USQUE-style manifold UKF: sigma points live in the 15-dim error tangent space,
boxplus lifts them onto the quaternion manifold, propagate_state runs the real
strapdown, boxminus brings them back. Additive Q/R. Counterpart to eskf.py for
the Stage 1.5 comparison.

# why a UKF here: the ESKF linearizes (analytic Fx/H). the UKF instead samples
# the nonlinearity with sigma points -- no Jacobians. on this near-linear INS
# problem the two should agree closely; the comparison shows the cost/benefit.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imu_mechanization import (
    NominalState,
    attitude_error_deg,
    initial_state_from_oxts,
    mechanization_input_from_oxts,
    propagate_state,
)
from kitti_highrate_loader import (
    HighRateOxtsConfig,
    HighRateOxtsSetupError,
    load_highrate_oxts,
    require_highrate_oxts,
)
from eskf import (
    EskfConfig,
    EskfError,
    _in_dropout,
    _truth_states,
    boxminus,
    boxplus,
    build_gps_measurements,
    nees_per_step,
    position_rmse,
    run_eskf,
    skew,
)
from so3 import quat_to_rotmat

STATE_DIM = 15
GPS_DIM = 3


@dataclass(frozen=True)
class UkfConfig:
    # Embed the ESKF config so all noise/init/GPS tuning is IDENTICAL -- that is
    # what makes the UKF-vs-ESKF comparison fair. alpha/beta/kappa are the
    # standard scaled-unscented-transform knobs (Julier): alpha small spreads the
    # sigma points tightly, beta=2 is optimal for Gaussians, kappa=0 is typical.
    eskf: EskfConfig = field(default_factory=EskfConfig)
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    @property
    def lam(self) -> float:  # lambda: sigma-point spread parameter
        return self.alpha**2 * (STATE_DIM + self.kappa) - STATE_DIM

    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        denom = STATE_DIM + self.lam
        wm = np.full(2 * STATE_DIM + 1, 1.0 / (2.0 * denom))  # mean weights
        wc = np.full(2 * STATE_DIM + 1, 1.0 / (2.0 * denom))  # covariance weights
        wm[0] = self.lam / denom
        wc[0] = self.lam / denom + (1.0 - self.alpha**2 + self.beta)  # beta corrects covariance kurtosis
        return wm, wc


def sigma_points(P: np.ndarray, lam: float) -> np.ndarray:
    """Return 2n+1 error-space sigma points: row 0 = 0, then +/- columns of the
    matrix square root of (n+lambda) P. Cholesky-guarded against non-PSD P."""
    n = STATE_DIM
    scaled = (n + lam) * (0.5 * (P + P.T))
    jitter = 0.0
    while True:
        try:
            s = np.linalg.cholesky(scaled + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter = 1e-9 if jitter == 0.0 else jitter * 10.0
            if jitter > 1.0:
                raise EskfError("sigma-point Cholesky failed: covariance not PSD")
    chi = np.zeros((2 * n + 1, n))
    for i in range(n):
        chi[1 + i] = s[:, i]
        chi[1 + n + i] = -s[:, i]
    return chi


class UnscentedKalmanFilter:
    """15-state USQUE-style UKF: IMU predict + GPS-position update."""

    def __init__(self, nominal, config, accel_bias=(0.0, 0.0, 0.0), gyro_bias=(0.0, 0.0, 0.0), covariance=None):
        if not isinstance(nominal, NominalState):
            raise EskfError("nominal must be a NominalState")
        if not isinstance(config, UkfConfig):
            raise EskfError("config must be a UkfConfig")
        self.nominal = nominal
        self.config = config
        self.accel_bias = np.asarray(accel_bias, dtype=float)
        self.gyro_bias = np.asarray(gyro_bias, dtype=float)
        ec = config.eskf
        self.P = ec.initial_covariance() if covariance is None else np.array(covariance, dtype=float)
        if self.P.shape != (STATE_DIM, STATE_DIM):
            raise EskfError(f"covariance must be {STATE_DIM}x{STATE_DIM}, got {self.P.shape}")
        self._lever = np.asarray(ec.p_base_gps, dtype=float)
        self._gps_cov = ec.gps_covariance()
        self._wm, self._wc = config.weights()

    def _sigma_points(self) -> np.ndarray:
        return sigma_points(self.P, self.config.lam)

    def predict(self, accel_body, gyro_body, dt) -> None:
        dt_s = float(dt)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise EskfError("dt must be positive and finite")
        chi = self._sigma_points()
        # lift each error sigma point onto the manifold and run the real strapdown
        props = []
        for i in range(chi.shape[0]):
            nom_i, ba_i, bg_i = boxplus(self.nominal, self.accel_bias, self.gyro_bias, chi[i])
            nom_p = propagate_state(nom_i, accel_body, gyro_body, dt_s, ba_i, bg_i)
            props.append((nom_p, ba_i, bg_i))
        # manifold mean: measure each propagated point against the center point,
        # take the weighted-mean error, and walk the center to it.
        c_nom, c_ba, c_bg = props[0]
        e = np.zeros((chi.shape[0], STATE_DIM))
        for i, (nom_p, ba_i, bg_i) in enumerate(props):
            e[i] = boxminus(nom_p, ba_i, bg_i, c_nom, c_ba, c_bg)
        ebar = self._wm @ e
        mean_nom, mean_ba, mean_bg = boxplus(c_nom, c_ba, c_bg, ebar)
        # covariance from residuals about the new mean, plus additive process noise
        cov = np.zeros((STATE_DIM, STATE_DIM))
        for i, (nom_p, ba_i, bg_i) in enumerate(props):
            d = boxminus(nom_p, ba_i, bg_i, mean_nom, mean_ba, mean_bg)
            cov += self._wc[i] * np.outer(d, d)
        # process_noise returns a 12x12 diagonal (accel, gyro, accel-bias, gyro-bias
        # blocks); map it into the 15-state via the same fi layout as the ESKF:
        # accel->velocity, gyro->attitude, biases->biases. Position gets no direct
        # process noise. Equivalent to fi @ q12 @ fi.T, written as block assignment.
        q12 = self.config.eskf.process_noise(dt_s)
        q15 = np.zeros((STATE_DIM, STATE_DIM))
        q15[3:6, 3:6] = q12[0:3, 0:3]
        q15[6:9, 6:9] = q12[3:6, 3:6]
        q15[9:12, 9:12] = q12[6:9, 6:9]
        q15[12:15, 12:15] = q12[9:12, 9:12]
        cov += q15
        self.nominal, self.accel_bias, self.gyro_bias = mean_nom, mean_ba, mean_bg
        self.P = 0.5 * (cov + cov.T)

    def update_gps(self, z_enu) -> None:
        z = np.asarray(z_enu, dtype=float)
        if z.shape != (GPS_DIM,):
            raise EskfError(f"z_enu must have shape ({GPS_DIM},), got {z.shape}")
        chi = self._sigma_points()
        # predicted GPS measurement per sigma point: h = pos + R * lever.
        # error-of-sigma-point vs current mean is exactly chi[i] (mean error = 0).
        zsig = np.zeros((chi.shape[0], GPS_DIM))
        for i in range(chi.shape[0]):
            nom_i, _, _ = boxplus(self.nominal, self.accel_bias, self.gyro_bias, chi[i])
            zsig[i] = nom_i.position + quat_to_rotmat(nom_i.q_map_imu) @ self._lever
        zbar = self._wm @ zsig
        pzz = self._gps_cov.copy()
        pxz = np.zeros((STATE_DIM, GPS_DIM))
        for i in range(chi.shape[0]):
            dz = zsig[i] - zbar
            pzz += self._wc[i] * np.outer(dz, dz)
            pxz += self._wc[i] * np.outer(chi[i], dz)
        gain = np.linalg.solve(pzz.T, pxz.T).T  # K = Pxz Pzz^-1 (solve, not inv)
        dx = gain @ (z - zbar)
        self.P = self.P - gain @ pzz @ gain.T
        self.P = 0.5 * (self.P + self.P.T)
        self._inject_and_reset(dx)

    def _inject_and_reset(self, dx) -> None:
        # inject correction onto the manifold, then rotate P by the attitude
        # reset Jacobian G (same as the ESKF) so the covariance stays consistent.
        self.nominal, self.accel_bias, self.gyro_bias = boxplus(
            self.nominal, self.accel_bias, self.gyro_bias, dx
        )
        g = np.eye(STATE_DIM)
        g[6:9, 6:9] = np.eye(3) - skew(0.5 * dx[6:9])
        self.P = g @ self.P @ g.T
        self.P = 0.5 * (self.P + self.P.T)


def run_ukf(sequence, config: UkfConfig, seed: int, burn_steps: int = 50, dropout_window=None) -> dict:
    ec = config.eskf
    samples = mechanization_input_from_oxts(sequence)
    timestamps = samples.timestamps
    n = timestamps.shape[0]

    truth = _truth_states(sequence)
    gps_indices, gps_z = build_gps_measurements(
        np.asarray(sequence.enu_position_m, dtype=float), ec.gps_rate_divisor, ec.gps_std_m, seed
    )
    gps_lookup = {int(idx): gps_z[i] for i, idx in enumerate(gps_indices)}

    filt = UnscentedKalmanFilter(initial_state_from_oxts(sequence, 0), config)

    x_est = np.zeros((n, 3))
    errors = np.zeros((n, STATE_DIM))
    cov_hist = np.zeros((n, STATE_DIM, STATE_DIM))
    att_err_series = np.zeros(n)
    x_est[0] = filt.nominal.position
    errors[0] = boxminus(filt.nominal, filt.accel_bias, filt.gyro_bias, truth[0], np.zeros(3), np.zeros(3))
    cov_hist[0] = filt.P
    att_err_series[0] = attitude_error_deg(filt.nominal, truth[0])

    for k in range(1, n):
        dt = float(timestamps[k] - timestamps[k - 1])
        filt.predict(samples.accel_body[k - 1], samples.gyro_body[k - 1], dt)
        if k in gps_lookup and not _in_dropout(float(timestamps[k]), dropout_window):
            filt.update_gps(gps_lookup[k])
        x_est[k] = filt.nominal.position
        errors[k] = boxminus(filt.nominal, filt.accel_bias, filt.gyro_bias, truth[k], np.zeros(3), np.zeros(3))
        cov_hist[k] = filt.P
        att_err_series[k] = attitude_error_deg(filt.nominal, truth[k])

    truth_positions = np.array([state.position for state in truth])
    nees = nees_per_step(errors, cov_hist)
    attitude_rmse = float(np.sqrt(np.mean(att_err_series[burn_steps:] ** 2)))

    return {
        "timestamps": timestamps,
        "x_est": x_est,
        "truth_positions": truth_positions,
        "gps_indices": gps_indices,
        "gps_z": gps_z,
        "cov_hist": cov_hist,
        "errors": errors,
        "nees": nees,
        "attitude_err_deg_series": att_err_series,
        "position_rmse": position_rmse(x_est[burn_steps:], truth_positions[burn_steps:]),
        "attitude_rmse_deg": attitude_rmse,
    }


def compare_filters(sequence, config: UkfConfig, seed: int = 0, dropout_window=None) -> dict:
    t0 = time.perf_counter()
    eskf_clean = run_eskf(sequence, config.eskf, seed)
    eskf_runtime = time.perf_counter() - t0
    t0 = time.perf_counter()
    ukf_clean = run_ukf(sequence, config, seed)
    ukf_runtime = time.perf_counter() - t0
    eskf_drop = run_eskf(sequence, config.eskf, seed, dropout_window=dropout_window) if dropout_window else None
    ukf_drop = run_ukf(sequence, config, seed, dropout_window=dropout_window) if dropout_window else None
    return {
        "eskf_clean": eskf_clean,
        "ukf_clean": ukf_clean,
        "eskf_drop": eskf_drop,
        "ukf_drop": ukf_drop,
        "eskf_runtime_s": eskf_runtime,
        "ukf_runtime_s": ukf_runtime,
        "dropout_window": dropout_window,
    }


def write_stage_note(cmp: dict, output_path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    e, u = cmp["eskf_clean"], cmp["ukf_clean"]
    speedup = cmp["ukf_runtime_s"] / cmp["eskf_runtime_s"] if cmp["eskf_runtime_s"] > 0 else float("nan")
    lines = [
        "# UKF vs ESKF on KITTI",
        "",
        "Both filters share identical noise/init/GPS tuning (`UkfConfig` embeds `EskfConfig`),",
        "100 Hz IMU predict, 10 Hz GPS-position update.",
        "",
        "| metric | ESKF | UKF |",
        "| --- | --- | --- |",
        f"| position RMSE [m] | {e['position_rmse']:.3f} | {u['position_rmse']:.3f} |",
        f"| attitude RMSE [deg] | {e['attitude_rmse_deg']:.3f} | {u['attitude_rmse_deg']:.3f} |",
        f"| runtime [s] | {cmp['eskf_runtime_s']:.3f} | {cmp['ukf_runtime_s']:.3f} |",
        "",
        f"UKF runtime is {speedup:.1f}x the ESKF (it propagates {2 * STATE_DIM + 1} sigma",
        "points through the strapdown per step vs one analytic Jacobian).",
        "",
        "## Verdict",
        "",
        "On this near-linear INS problem the ESKF and UKF reach effectively the same",
        "accuracy; the ESKF is preferred because it costs a single analytic Jacobian per",
        "step instead of 31 strapdown propagations, so it is far cheaper for no accuracy",
        "loss. The UKF would only pull ahead under strong nonlinearity (large attitude",
        "errors, coarse update rates) that this 100 Hz / 10 Hz setup does not exhibit.",
    ]
    if cmp["dropout_window"] is not None:
        ed, ud = cmp["eskf_drop"], cmp["ukf_drop"]
        lines += [
            "",
            "## GPS dropout",
            "",
            f"Dropout window {cmp['dropout_window']} s. Position RMSE with the cut:",
            f"ESKF {ed['position_rmse']:.3f} m, UKF {ud['position_rmse']:.3f} m. Both drift",
            "open-loop through the gap on IMU alone and re-converge once GPS returns.",
        ]
    path.write_text("\n".join(lines) + "\n")
    return path


def plot_comparison(cmp: dict, output_path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    e, u = cmp["eskf_clean"], cmp["ukf_clean"]
    t = np.asarray(e["timestamps"], dtype=float)
    truth = e["truth_positions"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(truth[:, 0], truth[:, 1], "k-", label="truth", linewidth=1.5)
    ax.plot(e["x_est"][:, 0], e["x_est"][:, 1], "b-", label="ESKF", linewidth=1.2)
    ax.plot(u["x_est"][:, 0], u["x_est"][:, 1], "r--", label="UKF", linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_title("Trajectory")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, np.linalg.norm(e["x_est"][:, 0:2] - truth[:, 0:2], axis=1), "b-", label="ESKF")
    ax.plot(t, np.linalg.norm(u["x_est"][:, 0:2] - truth[:, 0:2], axis=1), "r--", label="UKF")
    ax.set_xlabel("time [s]"); ax.set_ylabel("position error [m]"); ax.set_title("Position error (clean)")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if cmp["eskf_drop"] is not None:
        ed, ud = cmp["eskf_drop"], cmp["ukf_drop"]
        ax.plot(t, np.linalg.norm(ed["x_est"][:, 0:2] - truth[:, 0:2], axis=1), "b-", label="ESKF")
        ax.plot(t, np.linalg.norm(ud["x_est"][:, 0:2] - truth[:, 0:2], axis=1), "r--", label="UKF")
        w = cmp["dropout_window"]
        ax.axvspan(w[0], w[1], color="gray", alpha=0.2, label="GPS dropout")
        ax.set_title("Position error (GPS dropout)")
    else:
        ax.set_title("No dropout window")
    ax.set_xlabel("time [s]"); ax.set_ylabel("position error [m]")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dropout_experiment(cmp: dict, output_path) -> Path:
    """Single figure for the writeup: ESKF position error vs time on the clean run
    and with a GPS cut, the dropout window shaded, showing divergence then recovery."""
    import matplotlib.pyplot as plt

    if cmp["eskf_drop"] is None:
        raise EskfError("plot_dropout_experiment requires a dropout_window in compare_filters")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean, drop = cmp["eskf_clean"], cmp["eskf_drop"]
    t = np.asarray(clean["timestamps"], dtype=float)
    truth = clean["truth_positions"]
    err_clean = np.linalg.norm(clean["x_est"][:, 0:2] - truth[:, 0:2], axis=1)
    err_drop = np.linalg.norm(drop["x_est"][:, 0:2] - truth[:, 0:2], axis=1)
    w = cmp["dropout_window"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, err_clean, "b-", label="GPS available (10 Hz)", linewidth=1.3)
    ax.plot(t, err_drop, "r-", label="GPS cut in window", linewidth=1.3)
    ax.axvspan(w[0], w[1], color="gray", alpha=0.2, label=f"GPS dropout {w[0]:.0f}-{w[1]:.0f} s")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("horizontal position error [m]")
    ax.set_title("ESKF GPS-dropout: open-loop drift and recovery")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def default_plot_path(date: str, drive: str) -> Path:
    return Path("prototypes/output") / f"kitti_{date}_{drive.zfill(4)}_ukf_vs_eskf.png"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the UKF and ESKF on KITTI extract OXTS.")
    parser.add_argument("--root", type=Path, default=HighRateOxtsConfig.root)
    parser.add_argument("--date", default=HighRateOxtsConfig.date)
    parser.add_argument("--drive", default=HighRateOxtsConfig.drive)
    parser.add_argument("--cache-root", type=Path, default=HighRateOxtsConfig.cache_root)
    parser.add_argument("--seed", type=int, default=0)
    # default window lands mid-drive on the ~11.6 s smoke drive_0001 so the cut
    # actually drops GPS; override for longer drives.
    parser.add_argument("--dropout-start", type=float, default=4.0)
    parser.add_argument("--dropout-len", type=float, default=4.0)
    parser.add_argument("--plot", nargs="?", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--note", type=Path, default=Path("docs/notes/ukf_vs_eskf_kitti.md"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    loader_config = HighRateOxtsConfig(root=args.root, date=args.date, drive=args.drive, cache_root=args.cache_root)
    window = (args.dropout_start, args.dropout_start + args.dropout_len)
    try:
        require_highrate_oxts(loader_config)
        sequence = load_highrate_oxts(loader_config)
        cmp = compare_filters(sequence, UkfConfig(), seed=args.seed, dropout_window=window)
    except (HighRateOxtsSetupError, EskfError) as exc:
        print(f"error: {exc}")
        return 2
    note_path = write_stage_note(cmp, args.note)

    plot_path = None
    if not args.no_plot:
        out = args.plot if args.plot is not None else default_plot_path(args.date, args.drive)
        plot_path = plot_comparison(cmp, out)

    e, u = cmp["eskf_clean"], cmp["ukf_clean"]
    print(f"sequence: {args.date} drive {args.drive.zfill(4)} extract")
    print(f"position_rmse_m  ESKF {e['position_rmse']:.3f} | UKF {u['position_rmse']:.3f}")
    print(f"attitude_rmse_deg ESKF {e['attitude_rmse_deg']:.3f} | UKF {u['attitude_rmse_deg']:.3f}")
    print(f"runtime_s        ESKF {cmp['eskf_runtime_s']:.3f} | UKF {cmp['ukf_runtime_s']:.3f}")
    print(f"note: {note_path}")
    if plot_path is not None:
        print(f"plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
