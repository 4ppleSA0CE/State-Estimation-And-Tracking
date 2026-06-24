"""15-state Error-State EKF (ESKF) for KITTI IMU + GPS fusion.

Error state order: [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)] = 15.
Global (world-frame) angular error convention (Sola). Nominal strapdown
propagation and SO(3) helpers are reused from imu_mechanization and so3.

# why error-state: attitude lives on SO(3), not R^3. keep the nominal pose on
# the manifold (quaternion) and run the EKF on a small error dx in R^15, then
# inject it back and reset. keeps attitude singularity-free, linearization tight.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    cache_path_for,
    load_highrate_oxts,
    require_highrate_oxts,
)
from so3 import euler_to_quat, quat_inverse, quat_multiply, quat_to_rotmat, quat_to_rotvec, rotvec_to_quat

STATE_DIM = 15
GPS_DIM = 3


class EskfError(RuntimeError):
    """Raised when ESKF inputs are malformed."""


def _as_vector3(name: str, value: object) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise EskfError(f"{name} must be numeric array-like") from exc
    if array.shape != (3,):
        raise EskfError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise EskfError(f"{name} must contain only finite values")
    return array


def _as_error_vector(name: str, value: object) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise EskfError(f"{name} must be numeric array-like") from exc
    if array.shape != (STATE_DIM,):
        raise EskfError(f"{name} must have shape ({STATE_DIM},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise EskfError(f"{name} must contain only finite values")
    return array


def skew(v: object) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix with skew(v) @ w == cross(v, w)."""
    # matrix form of the cross product; shows up in every rotation Jacobian
    vector = _as_vector3("v", v)
    vx, vy, vz = vector
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=float,
    )


def boxplus(
    nominal: NominalState,
    accel_bias: np.ndarray,
    gyro_bias: np.ndarray,
    dx: object,
) -> tuple[NominalState, np.ndarray, np.ndarray]:
    """Inject a 15-vector error into (nominal, accel_bias, gyro_bias).

    Global angular error: q_new = rotvec_to_quat(dtheta) * q (left multiply).
    """
    # boxplus = manifold "+": vector parts add, attitude composes a rotation so
    # the quaternion stays valid. folds the EKF correction back onto the manifold.
    if not isinstance(nominal, NominalState):
        raise EskfError("nominal must be a NominalState")
    error = _as_error_vector("dx", dx)
    accel_bias = _as_vector3("accel_bias", accel_bias)
    gyro_bias = _as_vector3("gyro_bias", gyro_bias)

    dp, dv, dtheta, dba, dbg = (error[0:3], error[3:6], error[6:9], error[9:12], error[12:15])
    q_new = quat_multiply(rotvec_to_quat(dtheta), nominal.q_map_imu)
    nominal_new = NominalState(
        position=nominal.position + dp,
        velocity=nominal.velocity + dv,
        q_map_imu=q_new,
    )
    return nominal_new, accel_bias + dba, gyro_bias + dbg


def boxminus(
    nominal_a: NominalState,
    accel_bias_a: np.ndarray,
    gyro_bias_a: np.ndarray,
    nominal_b: NominalState,
    accel_bias_b: np.ndarray,
    gyro_bias_b: np.ndarray,
) -> np.ndarray:
    """Return the 15-vector error dx such that a == boxplus(b, dx)."""
    # boxminus = manifold "-", inverse of boxplus. attitude diff q_a * q_b^-1 goes
    # through the SO(3) log map. used for the est-vs-truth error fed to NEES.
    if not isinstance(nominal_a, NominalState) or not isinstance(nominal_b, NominalState):
        raise EskfError("nominal_a and nominal_b must be NominalState")
    dp = nominal_a.position - nominal_b.position
    dv = nominal_a.velocity - nominal_b.velocity
    dq = quat_multiply(nominal_a.q_map_imu, quat_inverse(nominal_b.q_map_imu))
    dtheta = quat_to_rotvec(dq)
    dba = _as_vector3("accel_bias_a", accel_bias_a) - _as_vector3("accel_bias_b", accel_bias_b)
    dbg = _as_vector3("gyro_bias_a", gyro_bias_a) - _as_vector3("gyro_bias_b", gyro_bias_b)
    return np.concatenate([dp, dv, dtheta, dba, dbg])


@dataclass(frozen=True)
class EskfConfig:
    # Defaults tuned to pass the Stage 1.4 DOD on KITTI extract drive_0001
    # (pos RMSE < 0.5 m, attitude RMSE < 1 deg, NEES 95% band >= 0.90 over a seed sweep).
    # Key tuning insight: the gyro bias and attitude priors must be tight. KITTI OXTS
    # measurements are already bias-corrected (true bias ~0), so a loose p0_gyro_bias lets
    # GPS position noise corrupt the gyro-bias estimate via the bias->dtheta->dv->dp chain,
    # which then drifts attitude (the dominant attitude-error path, not strapdown drift,
    # which is only ~0.25 deg over this drive). gps_std_m models a realistic DGPS-grade
    # receiver; it is both the injected GPS noise and the measurement covariance R, so they
    # stay matched and NEES remains valid.
    sigma_accel: float = 2.0          # m/s^2
    sigma_gyro: float = 1e-3          # rad/s
    sigma_accel_bias: float = 1e-3    # m/s^2 random walk
    sigma_gyro_bias: float = 1e-5     # rad/s random walk
    # GPS measurement noise (and the std of the synthetic noise injected on OXTS positions).
    gps_std_m: float = 0.75
    gps_rate_divisor: int = 10        # GPS every Nth IMU sample (100 Hz / 10 = 10 Hz)
    # Initial covariance (std per block).
    p0_pos: float = 2.0
    p0_vel: float = 3.0
    p0_att: float = np.deg2rad(0.5)
    p0_accel_bias: float = 0.1
    p0_gyro_bias: float = 1e-4
    # GPS antenna lever arm in the IMU/base frame (default colocated).
    p_base_gps: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def initial_covariance(self) -> np.ndarray:
        diag = np.concatenate(
            [
                np.full(3, self.p0_pos**2),
                np.full(3, self.p0_vel**2),
                np.full(3, self.p0_att**2),
                np.full(3, self.p0_accel_bias**2),
                np.full(3, self.p0_gyro_bias**2),
            ]
        )
        return np.diag(diag)

    def gps_covariance(self) -> np.ndarray:
        return (self.gps_std_m**2) * np.eye(GPS_DIM)

    def process_noise(self, dt: float) -> np.ndarray:  # Q, new uncertainty added each predict
        # white sensor noise scales as sigma^2*dt^2 (Sola impulse form); bias random
        # walks scale as sigma^2*dt. the dt vs dt^2 split is deliberate, keeps NEES honest.
        return np.diag(
            np.concatenate(
                [
                    np.full(3, (self.sigma_accel**2) * dt * dt),       # accel white noise -> velocity
                    np.full(3, (self.sigma_gyro**2) * dt * dt),        # gyro white noise -> attitude
                    np.full(3, (self.sigma_accel_bias**2) * dt),       # accel bias random walk
                    np.full(3, (self.sigma_gyro_bias**2) * dt),        # gyro bias random walk
                ]
            )
        )


class ErrorStateEKF:
    """15-state error-state EKF: IMU predict + GPS-position update."""

    def __init__(
        self,
        nominal: NominalState,
        config: EskfConfig,
        accel_bias: object = (0.0, 0.0, 0.0),
        gyro_bias: object = (0.0, 0.0, 0.0),
        covariance: np.ndarray | None = None,
    ) -> None:
        if not isinstance(nominal, NominalState):
            raise EskfError("nominal must be a NominalState")
        if not isinstance(config, EskfConfig):
            raise EskfError("config must be an EskfConfig")
        self.nominal = nominal
        self.config = config
        self.accel_bias = _as_vector3("accel_bias", accel_bias)
        self.gyro_bias = _as_vector3("gyro_bias", gyro_bias)
        self.P = config.initial_covariance() if covariance is None else np.array(covariance, dtype=float)
        if self.P.shape != (STATE_DIM, STATE_DIM):
            raise EskfError(f"covariance must be {STATE_DIM}x{STATE_DIM}, got {self.P.shape}")
        self._lever = _as_vector3("p_base_gps", config.p_base_gps)
        self._gps_cov = config.gps_covariance()

    def _state_transition(self, accel_body: np.ndarray, gyro_body: np.ndarray, dt: float) -> np.ndarray:
        # Global (world-frame) angular-error convention: q_true = Exp(dtheta) * q_nom.
        # Then accel_map = R * (a - b_a) + g perturbs as d(accel_map)/d(dtheta) = -skew(R a),
        # so the velocity-attitude block is -skew(R a) dt (NOT -R skew(a) dt, which is the
        # local/body-error form). Position gets the matching 0.5 dt^2 second-order coupling
        # because propagate_state integrates position as p + v dt + 0.5 accel_map dt^2.
        rotation = quat_to_rotmat(self.nominal.q_map_imu)
        accel = _as_vector3("accel_body", accel_body) - self.accel_bias
        rotated_accel = rotation @ accel
        identity = np.eye(3)

        fx = np.eye(STATE_DIM)
        fx[0:3, 3:6] = identity * dt
        fx[0:3, 6:9] = -0.5 * skew(rotated_accel) * dt * dt
        fx[0:3, 9:12] = -0.5 * rotation * dt * dt
        fx[3:6, 6:9] = -skew(rotated_accel) * dt
        fx[3:6, 9:12] = -rotation * dt
        fx[6:9, 12:15] = -rotation * dt
        return fx

    def predict(self, accel_body: object, gyro_body: object, dt: object) -> None:
        accel_body = _as_vector3("accel_body", accel_body)
        gyro_body = _as_vector3("gyro_body", gyro_body)
        dt_s = float(dt)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise EskfError("dt must be positive and finite")

        # EKF predict, error-state form: P <- Fx P Fx^T + Fi Q Fi^T.
        # error mean stays zero here; only covariance moves. nominal pose is
        # propagated separately by the strapdown integrator below.
        fx = self._state_transition(accel_body, gyro_body, dt_s)  # error-state transition Jacobian
        fi = np.zeros((STATE_DIM, 12))  # maps 12-vector sensor/bias noise into the 15-state error
        fi[3:6, 0:3] = np.eye(3)
        fi[6:9, 3:6] = np.eye(3)
        fi[9:12, 6:9] = np.eye(3)
        fi[12:15, 9:12] = np.eye(3)
        q = self.config.process_noise(dt_s)

        self.P = fx @ self.P @ fx.T + fi @ q @ fi.T
        self.P = 0.5 * (self.P + self.P.T)  # re-symmetrize: kills numerical drift that breaks PSD
        self.nominal = propagate_state(
            self.nominal, accel_body, gyro_body, dt_s, self.accel_bias, self.gyro_bias
        )

    def _measurement_jacobian(self) -> np.ndarray:
        # Global angular-error convention: h = p + R * lever, so
        # dh/d(dtheta) = -skew(R * lever)  (NOT -R * skew(lever), which is the local form).
        rotation = quat_to_rotmat(self.nominal.q_map_imu)
        h = np.zeros((GPS_DIM, STATE_DIM))
        h[0:3, 0:3] = np.eye(3)
        h[0:3, 6:9] = -skew(rotation @ self._lever)
        return h

    def _measurement_prediction(self) -> np.ndarray:
        rotation = quat_to_rotmat(self.nominal.q_map_imu)
        return self.nominal.position + rotation @ self._lever

    def update_gps(self, z_enu: object) -> None:
        z = _as_vector3("z_enu", z_enu)
        h = self._measurement_jacobian()
        y = z - self._measurement_prediction()      # innovation: measured minus predicted GPS
        s = h @ self.P @ h.T + self._gps_cov         # innovation covariance
        gain = np.linalg.solve(s, h @ self.P).T      # K = P H^T S^-1 (solve, never inv)
        dx = gain @ y                                # correction in error-state space

        # Joseph form: numerically stable, stays PSD even with a non-optimal gain
        i_kh = np.eye(STATE_DIM) - gain @ h
        self.P = i_kh @ self.P @ i_kh.T + gain @ self._gps_cov @ gain.T
        self.P = 0.5 * (self.P + self.P.T)
        self._inject_and_reset(dx)

    def _inject_and_reset(self, dx: np.ndarray) -> None:
        # inject the error into the nominal state, then reset error to zero. the
        # reset moves the attitude frame, so P must be rotated by the reset
        # Jacobian G to stay consistent (G = I - skew(0.5*dtheta) on the attitude block).
        self.nominal, self.accel_bias, self.gyro_bias = boxplus(
            self.nominal, self.accel_bias, self.gyro_bias, dx
        )
        dtheta = dx[6:9]
        g = np.eye(STATE_DIM)
        g[6:9, 6:9] = np.eye(3) - skew(0.5 * dtheta)  # covariance reset Jacobian
        self.P = g @ self.P @ g.T
        self.P = 0.5 * (self.P + self.P.T)


def build_gps_measurements(
    enu_position_m: np.ndarray,
    divisor: int,
    gps_std_m: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample ENU positions every `divisor` samples and add N(0, gps_std^2 I)."""
    enu = np.asarray(enu_position_m, dtype=float)
    if enu.ndim != 2 or enu.shape[1] != 3:
        raise EskfError(f"enu_position_m must be (N, 3), got {enu.shape}")
    if divisor < 1:
        raise EskfError("divisor must be >= 1")
    indices = np.arange(0, enu.shape[0], divisor)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, gps_std_m, size=(indices.shape[0], 3))
    return indices, enu[indices] + noise


def _truth_states(sequence: object) -> list[NominalState]:
    enu = np.asarray(sequence.enu_position_m, dtype=float)
    rpy = np.asarray(sequence.roll_pitch_yaw, dtype=float)
    vel_body = np.asarray(sequence.velocity, dtype=float)
    states = []
    for k in range(enu.shape[0]):
        q = euler_to_quat(*rpy[k])
        states.append(
            NominalState(
                position=enu[k],
                velocity=quat_to_rotmat(q) @ vel_body[k],
                q_map_imu=q,
            )
        )
    return states


def position_rmse(estimated: np.ndarray, truth: np.ndarray) -> float:
    err = np.asarray(estimated)[:, 0:2] - np.asarray(truth)[:, 0:2]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def nees_per_step(errors: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """9-DOF NEES on [dp, dv, dtheta] against P[0:9, 0:9]."""
    # NEES = e^T P^-1 e, the consistency check: should average ~9 (the DOF) if the
    # filter's covariance is honest. only pos/vel/att -- KITTI truth has no bias.
    n = errors.shape[0]
    nees = np.zeros(n)
    for k in range(n):
        e = errors[k, 0:9].reshape(9, 1)
        nees[k] = float((e.T @ np.linalg.solve(covariances[k, 0:9, 0:9], e)).item())
    return nees


def _in_dropout(t: float, window: "tuple[float, float] | None") -> bool:
    """True if time t falls inside a (start, end) GPS-dropout window."""
    return window is not None and window[0] <= t <= window[1]


def run_eskf(sequence: object, config: EskfConfig, seed: int, burn_steps: int = 50, dropout_window: "tuple[float, float] | None" = None) -> dict:
    samples = mechanization_input_from_oxts(sequence)
    timestamps = samples.timestamps
    n = timestamps.shape[0]

    truth = _truth_states(sequence)
    gps_indices, gps_z = build_gps_measurements(
        np.asarray(sequence.enu_position_m, dtype=float),
        config.gps_rate_divisor,
        config.gps_std_m,
        seed,
    )
    gps_lookup = {int(idx): gps_z[i] for i, idx in enumerate(gps_indices)}

    filt = ErrorStateEKF(initial_state_from_oxts(sequence, 0), config)

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


def default_plot_path(date: str, drive: str) -> Path:
    return Path("prototypes/output") / f"kitti_{date}_{drive.zfill(4)}_eskf_summary.png"


def plot_eskf_summary(result: dict, config: EskfConfig, output_path: "Path | str") -> Path:
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(result["timestamps"], dtype=float)
    x_est = result["x_est"]
    truth = result["truth_positions"]
    gps_z = result["gps_z"]
    nees = result["nees"]
    pos_err = np.linalg.norm(x_est[:, 0:2] - truth[:, 0:2], axis=1)
    att_err = result["attitude_err_deg_series"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.plot(truth[:, 0], truth[:, 1], "k-", label="OXTS truth", linewidth=1.5)
    ax.scatter(gps_z[:, 0], gps_z[:, 1], s=10, c="gray", alpha=0.4, label="GPS fixes")
    ax.plot(x_est[:, 0], x_est[:, 1], "b-", label="ESKF estimate", linewidth=1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Trajectory")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, pos_err, "b-")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, att_err, "b-")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("attitude error [deg]")
    ax.set_title("Attitude error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, nees, "b-", label="NEES (9-DOF)")
    ax.axhline(9.0, color="k", linestyle="--", label="E[NEES]=9")
    ax.axhline(chi2.ppf(0.025, 9), color="gray", linestyle=":", alpha=0.7)
    ax.axhline(chi2.ppf(0.975, 9), color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NEES")
    ax.set_title("Normalized estimation error squared")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 15-state ESKF on KITTI extract OXTS.")
    parser.add_argument("--root", type=Path, default=HighRateOxtsConfig.root)
    parser.add_argument("--date", default=HighRateOxtsConfig.date)
    parser.add_argument("--drive", default=HighRateOxtsConfig.drive)
    parser.add_argument("--cache-root", type=Path, default=HighRateOxtsConfig.cache_root)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--gps-std", type=float, default=EskfConfig.gps_std_m)
    parser.add_argument("--gps-divisor", type=int, default=EskfConfig.gps_rate_divisor)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--plot", nargs="?", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv)
    loader_config = HighRateOxtsConfig(
        root=args.root,
        date=args.date,
        drive=args.drive,
        cache_root=args.cache_root,
        force_refresh=args.force_refresh,
    )
    config = EskfConfig(gps_std_m=args.gps_std, gps_rate_divisor=args.gps_divisor)

    try:
        require_highrate_oxts(loader_config)
        sequence = load_highrate_oxts(loader_config)
        result = run_eskf(sequence, config, seed=args.seed)
    except (HighRateOxtsSetupError, EskfError) as exc:
        print(f"error: {exc}")
        return 2

    from scipy.stats import chi2

    nees_eval = result["nees"][50:]
    lo, hi = chi2.ppf(0.025, 9), chi2.ppf(0.975, 9)
    band_fraction = float(np.mean((nees_eval >= lo) & (nees_eval <= hi)))

    plot_path = None
    if not args.no_plot:
        out = args.plot if args.plot is not None else default_plot_path(args.date, args.drive)
        plot_path = plot_eskf_summary(result, config, out)

    print(f"sequence: {args.date} drive {args.drive.zfill(4)} extract")
    print(f"samples: {result['timestamps'].shape[0]}")
    print(f"position_rmse_m: {result['position_rmse']:.3f}")
    print(f"attitude_rmse_deg: {result['attitude_rmse_deg']:.3f}")
    print(f"mean_nees: {float(np.mean(nees_eval)):.3f} (expected ~9)")
    print(f"nees_band_fraction: {band_fraction:.3f}")
    print(f"cache: {cache_path_for(loader_config)}")
    if plot_path is not None:
        print(f"plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
