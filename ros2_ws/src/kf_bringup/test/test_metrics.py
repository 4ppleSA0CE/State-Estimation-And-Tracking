import numpy as np
import pytest

from kf_bringup.metrics import (
    AteResult, ate, chi2_band, nees, pair_by_stamp, percent_in_band, rpe,
)


# --- ATE ---------------------------------------------------------------------------------

def test_ate_zero_for_identical_trajectories():
    p = np.cumsum(np.ones((100, 3)), axis=0)
    r = ate(p, p)
    assert isinstance(r, AteResult)
    assert r.rmse == pytest.approx(0.0, abs=1e-12)
    assert r.max == pytest.approx(0.0, abs=1e-12)
    assert r.n == 100


def test_ate_constant_offset_equals_offset_norm():
    p = np.cumsum(np.ones((50, 3)), axis=0)
    r = ate(p + np.array([3.0, 4.0, 0.0]), p)          # offset norm 5
    assert r.rmse == pytest.approx(5.0, rel=1e-12)
    assert r.mean == pytest.approx(5.0, rel=1e-12)
    assert r.std == pytest.approx(0.0, abs=1e-12)
    assert r.sse == pytest.approx(50 * 25.0, rel=1e-12)


def test_ate_rmse_exceeds_mean_when_error_varies():
    # RMSE >= mean always, with equality only for constant error. A swapped implementation
    # (mean where rmse belongs) would break this.
    ref = np.zeros((3, 3))
    est = np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]])
    r = ate(est, ref)
    assert r.rmse > r.mean
    assert r.rmse == pytest.approx(np.sqrt((0 + 1 + 25) / 3))
    assert r.median == pytest.approx(1.0)


def test_ate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="shape mismatch"):
        ate(np.zeros((10, 3)), np.zeros((11, 3)))


def test_ate_rejects_nan():
    p = np.zeros((10, 3)); p[3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ate(p, np.zeros((10, 3)))


def test_ate_rejects_empty_and_wrong_width():
    with pytest.raises(ValueError):
        ate(np.zeros((0, 3)), np.zeros((0, 3)))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        ate(np.zeros((10, 2)), np.zeros((10, 2)))


# --- RPE ---------------------------------------------------------------------------------

def _straight(n=200, length=10.0):
    t = np.linspace(0, length, n)
    return np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)


def test_rpe_zero_for_identical_trajectories():
    p = _straight()
    assert rpe(p, p, delta_m=1.0).rmse == pytest.approx(0.0, abs=1e-12)


def test_rpe_ignores_constant_offset():
    """RPE is relative: translating the whole trajectory must not register."""
    p = _straight()
    assert rpe(p + np.array([7.0, -2.0, 1.0]), p, delta_m=1.0).rmse == pytest.approx(0.0, abs=1e-9)


def test_rpe_detects_scale_error():
    p = _straight()
    # 10% scale error over a 1 m segment leaves ~0.1 m of relative error.
    assert rpe(p * 1.1, p, delta_m=1.0).rmse == pytest.approx(0.1, rel=0.05)


def test_rpe_ate_disagree_on_pure_offset():
    """The two metrics must not be secretly the same function."""
    p = _straight()
    est = p + np.array([5.0, 0.0, 0.0])
    assert ate(est, p).rmse == pytest.approx(5.0)
    assert rpe(est, p, delta_m=1.0).rmse == pytest.approx(0.0, abs=1e-9)


def test_rpe_windows_are_measured_along_the_reference_not_the_estimate():
    """Segment endpoints must be chosen by walking the REFERENCE arc length.

    Caught by mutation testing: swapping in the estimate's arc length survived every other test
    here, because those cases differ only by an offset or a gentle 10% scale, which barely moves
    the endpoints. A 3x scale separates the two definitions decisively -- 10 m of reference is
    30 m of estimate, so the mutant selects segments a third as long and reports a third of the
    error.
    """
    ref = _straight(n=2000, length=100.0)
    est = ref * 3.0
    got = rpe(est, ref, delta_m=10.0).rmse
    assert got == pytest.approx(20.0, rel=0.02)          # |30 - 10| over a 10 m reference span
    assert got != pytest.approx(20.0 / 3.0, rel=0.2)     # what est-based windowing would give


def test_rpe_rejects_nonpositive_delta():
    p = _straight()
    with pytest.raises(ValueError, match="delta_m must be positive"):
        rpe(p, p, delta_m=0.0)


def test_rpe_raises_when_trajectory_shorter_than_delta():
    p = _straight(n=10, length=1.0)
    with pytest.raises(ValueError, match="shorter than delta_m"):
        rpe(p, p, delta_m=50.0)


# --- NEES ---------------------------------------------------------------------------------

def test_nees_matches_hand_computation():
    err = np.array([[1.0, 0.0], [0.0, 2.0]])
    cov = np.stack([np.eye(2), np.diag([1.0, 4.0])])
    got = nees(err, cov)
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(1.0)


def test_nees_uses_full_covariance_not_just_the_diagonal():
    """A correlated P gives a different answer than its diagonal; a diag-only bug would pass
    the identity test above but fail here."""
    p = np.array([[1.0, 0.9], [0.9, 1.0]])
    e = np.array([1.0, -1.0])
    got = nees(e[None, :], p[None, :, :])[0]
    assert got == pytest.approx(float(e @ np.linalg.solve(p, e)))
    assert got != pytest.approx(float(e @ np.linalg.solve(np.diag(np.diag(p)), e)))


def test_nees_returns_nan_for_singular_covariance():
    assert np.isnan(nees(np.array([[1.0, 0.0]]), np.zeros((1, 2, 2)))[0])


def test_nees_returns_nan_for_indefinite_covariance():
    """An indefinite P is not a valid covariance. solve() would return a finite, meaningless
    number here; Cholesky refuses, which is the point."""
    bad = np.array([[[1.0, 0.0], [0.0, -1.0]]])
    assert np.isnan(nees(np.array([[1.0, 1.0]]), bad)[0])


def test_nees_mean_is_near_dof_for_consistent_samples():
    rng = np.random.default_rng(0)
    n, d = 20000, 3
    p = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]])
    err = rng.multivariate_normal(np.zeros(d), p, size=n)
    got = nees(err, np.tile(p, (n, 1, 1)))
    assert np.mean(got) == pytest.approx(d, rel=0.05)


def test_nees_rejects_shape_mismatch():
    with pytest.raises(ValueError, match=r"cov must be"):
        nees(np.zeros((5, 3)), np.zeros((5, 2, 2)))


# --- chi-squared bands ----------------------------------------------------------------------

def test_chi2_band_brackets_the_dof():
    lo, hi = chi2_band(dof=3, n=1)
    assert lo < 3.0 < hi


def test_chi2_band_narrows_as_samples_grow():
    lo1, hi1 = chi2_band(dof=3, n=1)
    lo2, hi2 = chi2_band(dof=3, n=100)
    assert (hi2 - lo2) < (hi1 - lo1)
    assert lo2 < 3.0 < hi2


def test_chi2_band_rejects_bad_args():
    with pytest.raises(ValueError):
        chi2_band(dof=0, n=1)
    with pytest.raises(ValueError):
        chi2_band(dof=3, n=0)


def test_percent_in_band_is_about_95_for_consistent_samples():
    rng = np.random.default_rng(1)
    v = rng.chisquare(3, size=50000)
    assert percent_in_band(v, dof=3) == pytest.approx(95.0, abs=1.0)


def test_percent_in_band_ignores_nan_and_handles_all_nan():
    v = np.array([3.0, np.nan, 3.0])
    assert percent_in_band(v, dof=3) == pytest.approx(100.0)
    assert np.isnan(percent_in_band(np.array([np.nan, np.nan]), dof=3))


# --- stamp pairing --------------------------------------------------------------------------

def test_pair_by_stamp_recovers_the_offset_by_one_alignment():
    ref_t = np.arange(0, 1000, 10, dtype=np.int64)
    est_t = ref_t[1:]                                    # filter lags one sample
    idx = pair_by_stamp(est_t, ref_t)
    assert np.array_equal(idx, np.arange(1, len(ref_t)))


def test_pair_by_stamp_rejects_a_stamp_with_no_exact_match():
    ref_t = np.arange(0, 100, 10, dtype=np.int64)
    with pytest.raises(ValueError, match="no exact reference match"):
        pair_by_stamp(np.array([15], dtype=np.int64), ref_t)


def test_pair_by_stamp_rejects_stamp_past_the_end():
    ref_t = np.arange(0, 100, 10, dtype=np.int64)
    with pytest.raises(ValueError):
        pair_by_stamp(np.array([1000], dtype=np.int64), ref_t)
