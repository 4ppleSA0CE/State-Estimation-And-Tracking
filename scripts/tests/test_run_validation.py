"""Pure-logic tests for the multi-sequence validation sweep. No Docker, no network, no KITTI.

Everything here runs on synthetic arrays or on the frozen filter's own API, so a failure points
at run_validation.py and never at a missing dataset or a container that would not start.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "prototypes" / "python"))

import run_validation as rv  # noqa: E402
from eskf import ErrorStateEKF, EskfConfig  # noqa: E402
from imu_mechanization import NominalState  # noqa: E402
from so3 import euler_to_quat  # noqa: E402


# --- the sequence list ------------------------------------------------------------------

def test_sequence_list_has_no_duplicates():
    names = [s.name for s in rv.SEQUENCES]
    assert len(names) == len(set(names)), f"duplicate sequence(s) in SEQUENCES: {names}"


def test_sequence_list_has_no_derived_of_derived_entry():
    """Every drive must be one repair of a plain 4-digit KITTI drive.

    `0009_tail_clean` -- a repair of an already-derived slice -- would pass a naive
    "endswith _clean" check while silently measuring a hand-picked sub-window of a sub-window.
    """
    for s in rv.SEQUENCES:
        assert re.fullmatch(r"\d{4}_clean", s.drive), (
            f"{s.drive!r} is not a single repair of a 4-digit drive")


def test_sequence_list_dates_are_kitti_capture_dates():
    for s in rv.SEQUENCES:
        assert re.fullmatch(r"\d{4}_\d{2}_\d{2}", s.date), f"bad capture date {s.date!r}"


def test_sequence_cache_path_comes_from_the_frozen_loader():
    """The cache name must be built by kitti_highrate_loader.cache_path_for, never by string
    concatenation here -- a private copy silently forks the moment CACHE_VERSION moves."""
    from kitti_highrate_loader import CACHE_VERSION, HighRateOxtsConfig, cache_path_for

    s = rv.SEQUENCES[0]
    expect = cache_path_for(HighRateOxtsConfig(date=s.date, drive=s.drive,
                                               cache_root=rv.CACHE_DIR))
    assert rv.cache_path(s) == expect
    assert rv.cache_path(s).parent == rv.CACHE_DIR
    assert f"_v{CACHE_VERSION}.npz" in expect.name


# --- gps_only_ate -----------------------------------------------------------------------

def _straight_truth(n=30000):
    """A long, smooth trajectory: enough samples for the ATE to converge on its expectation."""
    t = np.linspace(0.0, 300.0, n)
    return np.stack([10.0 * t, 3.0 * np.sin(0.05 * t), 0.02 * t], axis=1)


def test_gps_only_ate_is_reproducible_for_a_fixed_seed():
    truth = _straight_truth(5000)
    a = rv.gps_only_ate(truth, gps_std_m=0.75, divisor=10, seed=0)
    b = rv.gps_only_ate(truth, gps_std_m=0.75, divisor=10, seed=0)
    assert a == b


def test_gps_only_ate_differs_across_seeds():
    """Reproducibility is worthless if the seed is being ignored entirely."""
    truth = _straight_truth(5000)
    assert rv.gps_only_ate(truth, seed=0) != rv.gps_only_ate(truth, seed=1)


def test_gps_only_ate_lands_near_sigma_root_three():
    """Three independent N(0, sigma^2) axes give E[|e|^2] = 3 sigma^2, so RMSE -> sigma*sqrt(3)."""
    truth = _straight_truth(30000)
    got = rv.gps_only_ate(truth, gps_std_m=0.75, divisor=10, seed=0)
    assert got == pytest.approx(0.75 * np.sqrt(3.0), rel=0.03)


def test_gps_only_ate_scales_with_the_noise_std():
    truth = _straight_truth(30000)
    assert rv.gps_only_ate(truth, gps_std_m=1.5, divisor=10, seed=0) == pytest.approx(
        2.0 * rv.gps_only_ate(truth, gps_std_m=0.75, divisor=10, seed=0), rel=0.05)


def test_gps_only_ate_honours_the_divisor():
    """A different divisor picks a different subset of samples, so it must move the number."""
    truth = _straight_truth(5000)
    assert rv.gps_only_ate(truth, divisor=10, seed=0) != rv.gps_only_ate(
        truth, divisor=7, seed=0)


def test_gps_only_ate_matches_the_pipelines_own_noise_synthesis():
    """Byte-for-byte the same fixes pipeline_replay publishes: indices first, then ONE
    rng.normal of shape (len(indices), 3). Any other call order re-orders the noise stream."""
    truth = _straight_truth(2000)
    idx = np.arange(0, len(truth), 10)
    z = truth[idx] + np.random.default_rng(0).normal(0.0, 0.75, (len(idx), 3))
    expect = float(np.sqrt(np.mean(np.sum((z - truth[idx]) ** 2, axis=1))))
    assert rv.gps_only_ate(truth, gps_std_m=0.75, divisor=10, seed=0) == pytest.approx(
        expect, abs=1e-12)


# --- the summary row builder ------------------------------------------------------------

def _row_values(**overrides):
    values = {c: 0.0 for c in rv.SUMMARY_COLUMNS}
    values["seq"] = "2011_09_26_0001_clean"
    values["n_samples"] = 1166
    values.update(overrides)
    return values


def test_summary_row_emits_exactly_the_declared_columns():
    row = rv.summary_row(_row_values())
    assert list(row) == list(rv.SUMMARY_COLUMNS)


def test_summary_columns_are_the_agreed_names_in_the_agreed_order():
    assert rv.SUMMARY_COLUMNS == (
        "seq", "n_samples", "duration_s", "path_len_m", "retention_percent",
        "max_intra_segment_gap_s", "ate_eskf_3d", "ate_eskf_horiz", "ate_ukf_3d",
        "ate_gps_only", "rpe_eskf", "nees_pct_in_band", "nis_pct_in_band",
        "runtime_ms_per_step",
    )


def test_summary_row_rejects_a_missing_column():
    values = _row_values()
    values.pop("rpe_eskf")
    with pytest.raises(ValueError, match="rpe_eskf"):
        rv.summary_row(values)


def test_summary_row_rejects_an_unknown_column():
    with pytest.raises(ValueError, match="ate_eskf_2d"):
        rv.summary_row(_row_values(ate_eskf_2d=1.0))


def test_summary_row_keeps_an_empty_cell_empty():
    """A repair column the CSV does not carry stays blank. Substituting 0.0 or 100.0 would
    invent a retention figure that nothing measured."""
    row = rv.summary_row(_row_values(retention_percent="", max_intra_segment_gap_s=""))
    assert row["retention_percent"] == ""
    assert row["max_intra_segment_gap_s"] == ""


# --- the private-method dependency guard ------------------------------------------------

def _filter(lever=(0.0, 0.0, 0.0)):
    nominal = NominalState(
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([0.5, -0.5, 0.1]),
        q_map_imu=euler_to_quat(0.1, -0.2, 0.3),
    )
    return ErrorStateEKF(nominal, EskfConfig(p_base_gps=lever))


def test_measurement_jacobian_is_three_by_fifteen():
    assert _filter()._measurement_jacobian().shape == (3, 15)


def test_measurement_jacobian_is_identity_and_zeros_for_the_default_lever_arm():
    """NIS is computed from these two private methods. If either changes shape or meaning the
    sweep's NIS column becomes silent nonsense, so the dependency is pinned here."""
    h = _filter()._measurement_jacobian()
    expect = np.zeros((3, 15))
    expect[0:3, 0:3] = np.eye(3)
    assert np.allclose(h, expect)


def test_measurement_jacobian_gains_an_attitude_block_once_the_lever_arm_is_nonzero():
    """Proves the guard above is asserting something real rather than a structural constant."""
    h = _filter(lever=(1.5, 0.0, 0.0))._measurement_jacobian()
    assert not np.allclose(h[0:3, 6:9], 0.0)


def test_measurement_prediction_is_the_nominal_position_for_the_default_lever_arm():
    f = _filter()
    assert np.allclose(f._measurement_prediction(), f.nominal.position)


# --- NIS ---------------------------------------------------------------------------------

def _synthetic_sequence(n=400, dt=0.01):
    """The minimum an OXTS sequence needs to drive the frozen mechanization: a stationary,
    level vehicle, so specific force is pure gravity reaction and gyro is zero."""
    from types import SimpleNamespace

    t = np.arange(n) * dt
    return SimpleNamespace(
        timestamps=t,
        enu_position_m=np.zeros((n, 3)),
        roll_pitch_yaw=np.zeros((n, 3)),
        velocity=np.zeros((n, 3)),
        accel_body=np.tile([0.0, 0.0, 9.81], (n, 1)),
        gyro_body=np.zeros((n, 3)),
    )


def test_nis_series_has_one_value_per_gps_update():
    seq = _synthetic_sequence(400)
    cfg = EskfConfig()
    nis = rv.nis_series(seq, cfg, seed=0)
    # build_gps_measurements takes index 0 too, but the filter only updates from k >= 1.
    assert len(nis) == len(np.arange(0, 400, cfg.gps_rate_divisor)) - 1


def test_nis_series_is_finite_and_nonnegative():
    nis = rv.nis_series(_synthetic_sequence(400), EskfConfig(), seed=0)
    assert np.isfinite(nis).all()
    assert (nis >= 0.0).all()


def test_nis_series_is_reproducible_for_a_fixed_seed():
    seq = _synthetic_sequence(400)
    a = rv.nis_series(seq, EskfConfig(), seed=0)
    b = rv.nis_series(seq, EskfConfig(), seed=0)
    assert np.array_equal(a, b)


def test_nis_of_a_stationary_truth_sits_near_three_degrees_of_freedom():
    """A 3-D innovation normalised by its own covariance averages the DOF when the filter is
    consistent. Far from 3 means R, P or the innovation sign convention is wrong."""
    nis = rv.nis_series(_synthetic_sequence(6000), EskfConfig(), seed=0)
    assert 1.0 < float(np.mean(nis[50:])) < 9.0


# --- small pure helpers ------------------------------------------------------------------

def test_path_length_of_a_straight_line_is_its_length():
    xyz = np.stack([np.linspace(0.0, 100.0, 501), np.zeros(501), np.zeros(501)], axis=1)
    assert rv.path_length_m(xyz) == pytest.approx(100.0)


def test_path_length_of_a_single_sample_is_zero():
    assert rv.path_length_m(np.zeros((1, 3))) == 0.0


def test_horizontal_ate_ignores_the_vertical_axis():
    ref = np.zeros((100, 3))
    est = np.tile([0.3, 0.4, 99.0], (100, 1))
    assert rv.horizontal_ate(est, ref) == pytest.approx(0.5)


def test_three_dimensional_and_horizontal_ate_differ_when_there_is_vertical_error():
    ref = np.zeros((100, 3))
    est = np.tile([0.3, 0.4, 1.2], (100, 1))
    from kf_bringup.metrics import ate

    assert ate(est, ref).rmse == pytest.approx(1.3)
    assert rv.horizontal_ate(est, ref) == pytest.approx(0.5)


# --- repair-table lookup -----------------------------------------------------------------

def _write_repair_csv(path: Path, header: str, row: str) -> Path:
    path.write_text(f"{header}\n{row}\n")
    return path


def test_repair_lookup_reads_retention_and_gap_when_present(tmp_path):
    csv = _write_repair_csv(
        tmp_path / "r.csv",
        "date,drive,clean_drive,retention_percent,max_intra_segment_gap_s",
        "2011_09_26,0009,0009_clean,58.932,0.010")
    got = rv.repair_lookup(csv)[("2011_09_26", "0009_clean")]
    assert got["retention_percent"] == "58.932"
    assert got["max_intra_segment_gap_s"] == "0.010"


def test_repair_lookup_yields_empty_cells_when_the_columns_are_absent(tmp_path):
    """The columns are owned by another script. Absent must mean blank, never a made-up 100."""
    csv = _write_repair_csv(tmp_path / "r.csv", "date,drive,clean_drive,n_kept",
                            "2011_09_26,0009,0009_clean,2758")
    got = rv.repair_lookup(csv)[("2011_09_26", "0009_clean")]
    assert got["retention_percent"] == ""
    assert got["max_intra_segment_gap_s"] == ""


def test_repair_lookup_of_a_missing_file_is_empty(tmp_path):
    assert rv.repair_lookup(tmp_path / "nope.csv") == {}


# --- LaTeX generation ---------------------------------------------------------------------

def test_latex_table_uses_booktabs_rules():
    tex = rv.latex_table([rv.summary_row(_row_values())])
    for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
        assert rule in tex


def test_latex_table_escapes_the_underscores_in_a_sequence_name():
    """`2011_09_26_0001_clean` in text mode is five subscript errors and a failed pdflatex."""
    tex = rv.latex_table([rv.summary_row(_row_values(seq="2011_09_26_0001_clean"))])
    assert "2011\\_09\\_26\\_0001\\_clean" in tex
    assert re.search(r"(?<!\\)_", tex.replace("\\_", "")) is None


def test_latex_table_has_one_body_row_per_sequence():
    rows = [rv.summary_row(_row_values(seq=f"drive_{i}")) for i in range(3)]
    body = rv.latex_table(rows).split("\\midrule")[1].split("\\bottomrule")[0]
    assert body.count("\\\\") == 3


def test_latex_table_renders_an_empty_cell_as_a_dash():
    """A blank cell in a LaTeX table row is invisible; an explicit dash says "not measured"."""
    tex = rv.latex_table([rv.summary_row(_row_values(retention_percent=""))])
    assert "--" in tex


def test_latex_table_survives_a_round_trip_through_the_summary_csv(tmp_path, monkeypatch):
    """--table-only regenerates the table from summary.csv, where every value is a STRING.
    The formatter has to cope with that, or the regenerated table silently differs from the
    one the sweep itself wrote."""
    import csv as _csv

    rows = [rv.summary_row(_row_values(seq="2011_09_26_0001_clean", retention_percent=""))]
    csv_path, tex_path = tmp_path / "summary.csv", tmp_path / "results_table.tex"
    monkeypatch.setattr(rv, "SUMMARY_CSV", csv_path)
    monkeypatch.setattr(rv, "LATEX_TABLE", tex_path)
    rv.write_summary_csv(rows, csv_path)

    assert rv.main(["--table-only"]) == 0
    with csv_path.open(newline="") as fh:
        reread = [rv.summary_row(dict(r)) for r in _csv.DictReader(fh)]
    assert tex_path.read_text() == rv.latex_table(reread)
    assert "2011\\_09\\_26\\_0001\\_clean" in tex_path.read_text()
