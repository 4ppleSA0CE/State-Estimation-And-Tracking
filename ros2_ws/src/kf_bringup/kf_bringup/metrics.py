"""Trajectory and consistency metrics. Pure functions over arrays -- no ROS, no I/O, no globals,
so the evaluator node, the sweep and the tests all compute identically by construction.

No trajectory alignment anywhere. Estimate and reference live in the same metric ENU frame with the
same origin; Umeyama alignment would absorb exactly the error being measured. evo is invoked
without -a for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class AteResult:
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    sse: float
    n: int


@dataclass(frozen=True)
class RpeResult:
    rmse: float
    mean: float
    n: int
    delta_m: float


def _check_pair(est: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est = np.asarray(est, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if est.shape != ref.shape:
        raise ValueError(f"shape mismatch: est {est.shape} vs ref {ref.shape}")
    if est.ndim != 2 or est.shape[1] != 3:
        raise ValueError(f"expected (N, 3) positions, got {est.shape}")
    if len(est) == 0:
        raise ValueError("empty trajectory")
    if not (np.isfinite(est).all() and np.isfinite(ref).all()):
        raise ValueError("non-finite values in trajectory")
    return est, ref


def ate(est: np.ndarray, ref: np.ndarray) -> AteResult:
    """Absolute trajectory error: per-sample Euclidean position error, unaligned."""
    est, ref = _check_pair(est, ref)
    e = np.linalg.norm(est - ref, axis=1)
    return AteResult(
        rmse=float(np.sqrt(np.mean(e ** 2))), mean=float(e.mean()),
        median=float(np.median(e)), std=float(e.std()),
        min=float(e.min()), max=float(e.max()),
        sse=float(np.sum(e ** 2)), n=int(len(e)),
    )


def rpe(est: np.ndarray, ref: np.ndarray, delta_m: float) -> RpeResult:
    """Relative pose error over segments of `delta_m` metres of reference path length.

    For each start index i, take the first j > i whose reference arc length from i reaches
    delta_m, and compare the estimate's displacement over [i, j] with the reference's. A rigid
    offset of the whole trajectory cancels; drift and scale error do not.
    """
    est, ref = _check_pair(est, ref)
    if delta_m <= 0:
        raise ValueError(f"delta_m must be positive, got {delta_m}")
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))])
    # searchsorted on the arc length: first index whose distance from s[i] reaches delta_m.
    j_of = np.searchsorted(s, s + delta_m, side="left")
    valid = j_of < len(ref)
    if not valid.any():
        raise ValueError(f"trajectory shorter than delta_m={delta_m} m (total path {s[-1]:.2f} m)")
    i_idx = np.flatnonzero(valid)
    j_idx = j_of[valid]
    e = np.linalg.norm((est[j_idx] - est[i_idx]) - (ref[j_idx] - ref[i_idx]), axis=1)
    return RpeResult(rmse=float(np.sqrt(np.mean(e ** 2))), mean=float(e.mean()),
                     n=int(len(e)), delta_m=float(delta_m))


def nees(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-step e^T P^-1 e. NaN where P is not positive definite -- never a garbage number."""
    err = np.asarray(err, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if err.ndim != 2:
        raise ValueError(f"err must be (N, d), got {err.shape}")
    if cov.shape != (err.shape[0], err.shape[1], err.shape[1]):
        raise ValueError(f"cov must be (N, d, d) for err {err.shape}, got {cov.shape}")
    out = np.full(len(err), np.nan)
    for k, (e, p) in enumerate(zip(err, cov)):
        if not (np.isfinite(e).all() and np.isfinite(p).all()):
            continue
        try:
            # Cholesky first: it fails loudly on a non-PD matrix, where solve() would happily
            # return a finite but meaningless number for an indefinite P.
            chol = np.linalg.cholesky(p)
        except np.linalg.LinAlgError:
            continue
        y = np.linalg.solve(chol, e)
        out[k] = float(y @ y)
    return out


def chi2_band(dof: int, n: int = 1, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided (1-alpha) acceptance band for the mean of n chi-squared(dof) samples."""
    if dof < 1 or n < 1:
        raise ValueError(f"dof and n must be >= 1, got dof={dof}, n={n}")
    return (float(chi2.ppf(alpha / 2, dof * n) / n),
            float(chi2.ppf(1 - alpha / 2, dof * n) / n))


def percent_in_band(values: np.ndarray, dof: int, alpha: float = 0.05) -> float:
    """Percentage of per-step NEES/NIS values inside the single-sample chi-squared band."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    lo, hi = chi2_band(dof, n=1, alpha=alpha)
    return float(np.mean((v >= lo) & (v <= hi)) * 100.0)


def pair_by_stamp(est_t_ns: np.ndarray, ref_t_ns: np.ndarray) -> np.ndarray:
    """Index into the reference for each estimate stamp, requiring an EXACT match.

    The filter publishes the state for t_k only when the IMU sample at t_{k+1} arrives, so the
    estimate series is one sample shorter than the truth series and starts at truth index 1.
    Pairing by position instead of by stamp shifts the whole trajectory by one sample, which
    silently inflates ATE -- measured at 1.33 m vs 1.22 m max error on a recorded run.
    """
    est_t_ns = np.asarray(est_t_ns)
    ref_t_ns = np.asarray(ref_t_ns)
    if est_t_ns.ndim != 1 or ref_t_ns.ndim != 1:
        raise ValueError("timestamp arrays must be 1-D")
    if len(est_t_ns) == 0:
        raise ValueError("empty estimate timestamps")
    idx = np.clip(np.searchsorted(ref_t_ns, est_t_ns), 0, len(ref_t_ns) - 1)
    if not np.array_equal(ref_t_ns[idx], est_t_ns):
        bad = int((ref_t_ns[idx] != est_t_ns).sum())
        raise ValueError(f"{bad} of {len(est_t_ns)} estimate stamps have no exact reference match")
    return idx
