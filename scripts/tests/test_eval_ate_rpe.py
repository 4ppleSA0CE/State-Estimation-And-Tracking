import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_ate_rpe import TOL, cross_check, run_evo_ape, run_evo_rpe, write_tum  # noqa: E402

HAVE_EVO = subprocess.run(["which", "evo_ape"], capture_output=True).returncode == 0
needs_evo = pytest.mark.skipif(not HAVE_EVO, reason="evo not installed")


def _traj(n=300):
    t = np.linspace(0, 30, n)
    p = np.stack([t, 0.5 * np.sin(t), np.zeros_like(t)], axis=1)
    return t, p


# --- TUM writing (no evo needed) -------------------------------------------------------

def test_write_tum_shape_and_columns(tmp_path):
    t, p = _traj(10)
    rows = [line.split() for line in write_tum(tmp_path / "a.tum", t, p).read_text().splitlines()]
    assert len(rows) == 10
    assert all(len(r) == 8 for r in rows)


def test_write_tum_defaults_to_identity_quaternion(tmp_path):
    t, p = _traj(5)
    rows = np.loadtxt(write_tum(tmp_path / "a.tum", t, p))
    assert np.allclose(rows[:, 4:], np.tile([0.0, 0.0, 0.0, 1.0], (5, 1)))


def test_write_tum_rejects_length_mismatch(tmp_path):
    t, p = _traj(10)
    with pytest.raises(ValueError, match="timestamps for"):
        write_tum(tmp_path / "a.tum", t[:9], p)


def test_write_tum_rejects_bad_quaternion_shape(tmp_path):
    t, p = _traj(10)
    with pytest.raises(ValueError, match="quaternions"):
        write_tum(tmp_path / "a.tum", t, p, np.zeros((10, 3)))


# --- agreement with evo ----------------------------------------------------------------

@needs_evo
def test_evo_ape_matches_the_analytic_offset(tmp_path):
    t, p = _traj()
    off = np.array([0.3, -0.2, 0.05])
    ref_f = write_tum(tmp_path / "ref.tum", t, p)
    est_f = write_tum(tmp_path / "est.tum", t, p + off)
    assert run_evo_ape(ref_f, est_f) == pytest.approx(float(np.linalg.norm(off)), abs=1e-6)


@needs_evo
def test_evo_rpe_is_zero_for_a_rigid_offset(tmp_path):
    """RPE is relative, so a constant offset must not register -- in evo's implementation too."""
    t, p = _traj(600)
    ref_f = write_tum(tmp_path / "ref.tum", t, p)
    est_f = write_tum(tmp_path / "est.tum", t, p + np.array([2.0, -1.0, 0.5]))
    assert run_evo_rpe(ref_f, est_f, delta_m=5.0) == pytest.approx(0.0, abs=1e-6)


@needs_evo
def test_cross_check_passes_when_implementations_agree(tmp_path):
    t, p = _traj()
    res = cross_check(t, p + np.array([0.3, -0.2, 0.05]), p, tmp_path)
    assert abs(res["ate_rmse"] - res["evo_ate_rmse"]) < TOL


@needs_evo
def test_cross_check_fails_on_a_seeded_one_metre_shift(tmp_path):
    """The gate must be able to fail. Corrupt only the trajectory evo reads."""
    t, p = _traj()
    with pytest.raises(AssertionError, match="ATE disagreement"):
        cross_check(t, p + np.array([0.3, -0.2, 0.05]), p, tmp_path, _seed_shift_m=1.0)


@needs_evo
def test_cross_check_catches_a_shift_far_below_one_metre(tmp_path):
    """Guards the tolerance itself: a 1e-4 m corruption is still 100x above TOL."""
    t, p = _traj()
    with pytest.raises(AssertionError):
        cross_check(t, p, p, tmp_path, _seed_shift_m=1e-4)


@needs_evo
def test_evo_is_invoked_without_alignment(tmp_path):
    """A pure offset must survive as ATE. If -a (Umeyama) were passed, evo would absorb the
    offset and report ~0, which is exactly the failure this whole gate exists to prevent."""
    t, p = _traj()
    ref_f = write_tum(tmp_path / "ref.tum", t, p)
    est_f = write_tum(tmp_path / "est.tum", t, p + np.array([5.0, 0.0, 0.0]))
    assert run_evo_ape(ref_f, est_f) == pytest.approx(5.0, abs=1e-6)
