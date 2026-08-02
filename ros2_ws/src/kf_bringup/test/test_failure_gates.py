"""Host-side tests for the failure-mode gate evaluator. Fixtures are hand-built arrays -- these
tests must never need a recorded run, or a gate regression hides behind missing data.

Run from the repo root:
    python3 -m pytest ros2_ws/src/kf_bringup/test/test_failure_gates.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kf_bringup import failure_gates  # noqa: E402
from kf_bringup.failure_gates import (  # noqa: E402
    COAST_REACQUIRE_MAX_M, COUPLING_MIN_FRAMES, MODES,
    _across_gap, _coupling_r, confirmed_track_ids, ego_error, evaluate, id_switches,
    match_tracks, track_error_series,
)

N_TARGETS = 4          # kept local on purpose: this suite must not import targets.py
BOX_DIM = 7
N_IMM_MODES = 4        # CV, CA, CT+, CT-
CAR_DIMS = (3.9, 1.6, 1.5)
GROUND_Z = -1.7

# Targets 15 m apart, so a 3 m association gate is never ambiguous by accident.
DEFAULT_TARGET_XY = np.array([[20.0, 0.0], [35.0, 0.0], [50.0, 0.0], [65.0, 0.0]])
FAR_AWAY = (1000.0, 1000.0)     # where extra (clutter-born) tracks are parked


# ---------------------------------------------------------------------------
# Fixture builder -- every array of the recorded-pipeline npz schema, built by hand.
# ---------------------------------------------------------------------------

def _make_run(
    *,
    n_oxts: int = 1200,
    oxts_dt: float = 0.01,
    frame_every: int = 10,
    ego_err=0.0,                 # scalar or (n_oxts,) horizontal error, applied along east
    ego_est_ns=None,             # explicit /ego/state stamps (subset of t_ns, any order)
    target_xy=None,              # (T, 2) constant ENU placement
    visible=None,                # (F, T) bool
    track_ids=None,              # (F, K) int
    track_xy=None,               # (F, K, 2) ENU
    track_off=0.0,               # scalar, (F,) or (F, T): east offset of each track from truth
    track_count=None,            # (F,) int
    track_mode=None,             # (F, K, B)
    accel_bias_x: float = 0.0,
    imu_bias_xyz=(0.0, 0.0, 0.0),
    gps_window=(0.0, 0.0),
    det_window=(0.0, 0.0),
    mode: str = "baseline",
    params=None,
) -> dict:
    t = np.arange(n_oxts, dtype=float) * oxts_dt
    t_ns = np.round(t * 1e9).astype(np.int64)

    err = np.asarray(ego_err, dtype=float)
    err = np.full(n_oxts, float(err)) if err.ndim == 0 else err.astype(float)
    assert err.shape == (n_oxts,)
    ego_truth = np.zeros((n_oxts, 3))
    ego_est_full = np.zeros((n_oxts, 3))
    ego_est_full[:, 0] = err                      # error is purely along east -> |err|

    if ego_est_ns is None:
        est_ns, est_pos = t_ns.copy(), ego_est_full.copy()
    else:
        est_ns = np.asarray(ego_est_ns, dtype=np.int64)
        row_of = {int(v): i for i, v in enumerate(t_ns.tolist())}
        rows = [row_of[int(v)] for v in est_ns.tolist()]
        est_pos = ego_est_full[rows]

    n_est = est_ns.size
    accel_bias = np.zeros((n_est, 3))
    accel_bias[:, 0] = accel_bias_x

    fidx = np.arange(0, n_oxts, frame_every)
    frame_t, frame_t_ns = t[fidx], t_ns[fidx]
    n_frames = fidx.size

    tgt_xy = DEFAULT_TARGET_XY if target_xy is None else np.asarray(target_xy, dtype=float)
    n_targets = tgt_xy.shape[0]
    truth = np.zeros((n_frames, n_targets, BOX_DIM))
    truth[:, :, 0] = tgt_xy[:, 0]
    truth[:, :, 1] = tgt_xy[:, 1]
    truth[:, :, 2] = GROUND_Z
    truth[:, :, 4], truth[:, :, 5], truth[:, :, 6] = CAR_DIMS

    vis = np.ones((n_frames, n_targets), dtype=bool) if visible is None \
        else np.asarray(visible, dtype=bool)

    ids = np.tile(10 + np.arange(n_targets, dtype=np.int64), (n_frames, 1)) \
        if track_ids is None else np.asarray(track_ids, dtype=np.int64)
    n_slots = ids.shape[1]

    if track_xy is None:
        off = np.asarray(track_off, dtype=float)
        if off.ndim == 0:
            off = np.full((n_frames, n_targets), float(off))
        elif off.ndim == 1:
            off = np.repeat(off.reshape(n_frames, 1), n_targets, axis=1)
        xy = np.zeros((n_frames, n_slots, 2))
        xy[:, :, 0] = FAR_AWAY[0]
        xy[:, :, 1] = FAR_AWAY[1]
        share = min(n_slots, n_targets)
        xy[:, :share, 0] = tgt_xy[:share, 0] + off[:, :share]
        xy[:, :share, 1] = tgt_xy[:share, 1]
    else:
        xy = np.asarray(track_xy, dtype=float)

    counts = np.full(n_frames, n_slots, dtype=np.int32) if track_count is None \
        else np.asarray(track_count, dtype=np.int32)

    pos_enu = np.full((n_frames, n_slots, 3), np.nan)
    state = np.full((n_frames, n_slots, 4), np.nan)
    for k in range(n_frames):
        for j in range(int(counts[k])):
            pos_enu[k, j] = (xy[k, j, 0], xy[k, j, 1], GROUND_Z)
            state[k, j] = (xy[k, j, 0], xy[k, j, 1], 0.0, 0.0)

    if track_mode is None:
        modes = np.full((n_frames, n_slots, N_IMM_MODES), np.nan)
        for k in range(n_frames):
            for j in range(int(counts[k])):
                modes[k, j] = (1.0, 0.0, 0.0, 0.0)      # pure CV
    else:
        modes = np.asarray(track_mode, dtype=float)

    max_det = max(1, n_targets)
    return {
        "mode": mode,
        "params_json": json.dumps({} if params is None else params),
        "t": t, "t_ns": t_ns,
        "ego_truth": ego_truth,
        "ego_est": est_pos, "ego_est_t_ns": est_ns,
        "ego_accel_bias": accel_bias, "ego_gyro_bias": np.zeros((n_est, 3)),
        "frame_t": frame_t, "frame_t_ns": frame_t_ns,
        "target_truth_enu": truth, "target_visible": vis,
        "det_count": np.zeros(n_frames, dtype=np.int32),
        "det_boxes": np.full((n_frames, max_det, BOX_DIM), np.nan),
        "det_src_id": np.full((n_frames, max_det), -1, dtype=np.int32),
        "track_count": counts, "track_ids": ids.astype(np.int32),
        "track_state": state, "track_pos_enu": pos_enu, "track_mode": modes,
        "gps_window": np.asarray(gps_window, dtype=float),
        "det_window": np.asarray(det_window, dtype=float),
        "imu_bias_xyz": np.asarray(imu_bias_xyz, dtype=float),
    }


def _fails(lines):
    return [ln for ln in lines if ln.startswith("[FAIL]")]


# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------

# Every number below is TRANSCRIBED BY HAND from design doc section 6 (and section 5.4 for the
# association gate). Not one is computed from the module under test -- a threshold derived from
# the code it is meant to pin cannot detect the code drifting, which is exactly how eight of
# these previously survived being retuned with the suite still green.
SPEC_THRESHOLDS = {
    # section 5.4: "greedy nearest-neighbour in the ENU ground plane with a 3.0 m cutoff"
    "MATCH_GATE_M": 3.0,
    # baseline row: "ego_rmse < 1.0 m ... track_rmse < 2.0 m"
    "BASELINE_EGO_RMSE_MAX": 1.0,
    "BASELINE_TRACK_RMSE_MAX": 2.0,
    # gps_dropout row: "(a) peak ego error in-window >= 3x baseline peak; (b) ego error <= 2x
    # baseline RMSE within 2 s after GPS returns; (c) peak in-window track ENU error >= 2x
    # baseline track RMSE, and r > 0.8"; the correlation "needs at least 20 usable frames"
    "DROPOUT_EGO_PEAK_RATIO": 3.0,
    "DROPOUT_RECOVERY_RATIO": 2.0,
    "DROPOUT_RECOVERY_S": 2.0,
    "DROPOUT_TRACK_PEAK_RATIO": 2.0,
    "COUPLING_R_MIN": 0.8,
    "COUPLING_MIN_FRAMES": 20,
    # imu_bias row: "|b_x| >= 0.3 x 0.1 at the final sample"
    "IMU_BIAS_MIN_FRACTION": 0.3,
    # maneuver row: "CT mode probability > 0.5 within 20 frames of onset, and its CV
    # probability drops below 0.3 at the CT peak"
    "MANEUVER_CT_MIN": 0.5,
    "MANEUVER_CV_MAX": 0.3,
    "MANEUVER_MAX_FRAMES": 20,
    # det_dropout_coast row: "position error at re-acquisition < 5 m"
    "COAST_REACQUIRE_MAX_M": 5.0,
    # clutter row: "confirmed-track count <= baseline + 2; ID switches <= baseline + 2"
    "CLUTTER_SLACK": 2,
}


def test_pre_registered_thresholds_match_the_spec():
    """All 15 thresholds are pre-registered ceilings, so retuning one is a spec change, not a
    tuning knob. This test is what makes that true: it fails on any silent drift."""
    assert len(SPEC_THRESHOLDS) == 15
    got = {name: getattr(failure_gates, name) for name in SPEC_THRESHOLDS}
    assert got == SPEC_THRESHOLDS

    # And no NEW threshold may be added to the module without being pinned here too.
    exported = {n for n in failure_gates.__all__ if n.isupper() and n != "MODES"}
    assert exported - {"MODE_CV", "MODE_CA", "FIRST_CT_MODE", "X_ENU", "Y_ENU"} \
        == set(SPEC_THRESHOLDS)


# ---------------------------------------------------------------------------
# ego_error
# ---------------------------------------------------------------------------

def test_ego_error_aligns_by_exact_nanosecond_stamp():
    n = 300
    err = 0.001 * np.arange(n, dtype=float)          # every sample has a distinct error
    run = _make_run(n_oxts=n, ego_err=err)
    picked = [201, 3, 299, 100, 57]                  # a shuffled subset, not sorted
    run["ego_est_t_ns"] = run["t_ns"][picked].copy()
    run["ego_est"] = run["ego_est"][picked].copy()

    stamps, got = ego_error(run)
    assert stamps.tolist() == run["t_ns"][picked].tolist()
    assert got == pytest.approx(err[picked])         # order preserved, no accidental sort


def test_ego_error_rejects_a_stamp_with_no_truth_match():
    run = _make_run(n_oxts=50, ego_err=0.2)
    run["ego_est_t_ns"] = run["ego_est_t_ns"].copy()
    run["ego_est_t_ns"][7] += 1                      # one nanosecond off a real truth stamp
    with pytest.raises(ValueError, match="exact-stamp only"):
        ego_error(run)


# ---------------------------------------------------------------------------
# match_tracks
# ---------------------------------------------------------------------------

def test_match_tracks_pairs_nearest_within_the_gate():
    run = _make_run(
        n_oxts=10, frame_every=10,
        track_ids=np.array([[77]]),
        track_xy=np.array([[[DEFAULT_TARGET_XY[2, 0] + 1.0, 0.0]]]),
        track_count=np.array([1]),
    )
    assert match_tracks(run, 0) == {2: (77, pytest.approx(1.0))}


# 3.0 is written as a LITERAL, never as MATCH_GATE_M: a parameter read from the module under
# test follows the constant when it drifts, so the boundary case would silently move with it.
@pytest.mark.parametrize("distance,matched", [(2.999, True), (3.0, False), (5.0, False)])
def test_match_tracks_rejects_beyond_the_gate(distance, matched):
    run = _make_run(
        n_oxts=10, frame_every=10,
        track_ids=np.array([[77]]),
        track_xy=np.array([[[DEFAULT_TARGET_XY[2, 0] + distance, 0.0]]]),
        track_count=np.array([1]),
    )
    assert (2 in match_tracks(run, 0)) is matched


def test_match_tracks_never_reuses_one_track_for_two_targets():
    run = _make_run(
        n_oxts=10, frame_every=10,
        target_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
        track_ids=np.array([[42]]),
        track_xy=np.array([[[0.5, 0.0]]]),
        track_count=np.array([1]),
    )
    m = match_tracks(run, 0)
    assert len(m) == 1 and m == {0: (42, pytest.approx(0.5))}


def test_match_tracks_skips_invisible_targets():
    kwargs = dict(n_oxts=10, frame_every=10, track_off=0.4)
    assert 2 in match_tracks(_make_run(**kwargs), 0)               # control
    hidden = np.ones((1, N_TARGETS), dtype=bool)
    hidden[0, 2] = False
    m = match_tracks(_make_run(visible=hidden, **kwargs), 0)
    assert 2 not in m and set(m) == {0, 1, 3}


# ---------------------------------------------------------------------------
# id_switches
# ---------------------------------------------------------------------------

def test_id_switch_count_detects_a_changed_track_id():
    n_frames = 10
    ids = np.full((n_frames, 1), 5, dtype=np.int64)
    ids[5:, 0] = 9
    run = _make_run(
        n_oxts=n_frames * 10, target_xy=np.array([[20.0, 0.0]]),
        track_ids=ids, track_count=np.ones(n_frames, dtype=int),
    )
    assert id_switches(run) == 1


def test_id_switch_ignores_gaps_where_the_target_is_unmatched():
    n_frames = 10
    ids = np.full((n_frames, 1), 5, dtype=np.int64)
    ids[6:, 0] = 9
    counts = np.ones(n_frames, dtype=int)
    counts[4:6] = 0                                  # the target is unmatched for two frames
    run = _make_run(
        n_oxts=n_frames * 10, target_xy=np.array([[20.0, 0.0]]),
        track_ids=ids, track_count=counts,
    )
    assert id_switches(run) == 0                     # a gap must never fabricate a switch
    _err, cnt = track_error_series(run)
    assert cnt.tolist() == [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# baseline gate
# ---------------------------------------------------------------------------

def test_baseline_gate_passes_a_clean_run():
    run = _make_run(ego_err=0.2, track_off=0.5)
    passed, lines = evaluate("baseline", run)
    assert passed, lines
    text = "\n".join(lines)
    assert "ego_rmse = 0.2000 m" in text            # the measured value, not just a verdict
    assert "track_rmse = 0.5000 m" in text
    assert "4/4 targets confirmed" in text


def test_baseline_gate_fails_on_large_ego_error():
    passed, lines = evaluate("baseline", _make_run(ego_err=2.0, track_off=0.5))
    assert not passed
    assert any("ego_rmse = 2.0000 m" in ln for ln in _fails(lines))


def test_baseline_gate_fails_when_a_target_is_never_confirmed():
    n_frames = 12
    run = _make_run(
        n_oxts=n_frames * 10, track_off=0.5,
        track_ids=np.tile(np.array([10, 11, 12], dtype=np.int64), (n_frames, 1)),
        track_count=np.full(n_frames, 3),
    )
    passed, lines = evaluate("baseline", run)
    assert not passed
    assert any("never confirmed: [3]" in ln for ln in _fails(lines))


# ---------------------------------------------------------------------------
# gps_dropout gate
# ---------------------------------------------------------------------------

LO, HI = 4.0, 8.0


def _dropout_pair(broken=None, frame_every=10, n_oxts=1200, oxts_dt=0.01):
    """(run, baseline). Ego error and track error are both affine in the same in-window ramp,
    so the intact pair correlates at r = 1; `broken` disables exactly one condition."""
    t = np.arange(n_oxts, dtype=float) * oxts_dt
    ramp = np.clip((t - LO) / (HI - LO), 0.0, 1.0)
    scale = 0.1 if broken == "a" else 1.9           # (a): in-window peak stays under 3x baseline
    ego = 0.1 + scale * ramp
    if broken != "b":                                # (b): leave the error high after recovery
        ego = np.where(t > HI, 0.1, ego)

    frames = t[np.arange(0, n_oxts, frame_every)]
    framp = np.clip((frames - LO) / (HI - LO), 0.0, 1.0)
    if broken == "c":                                # varies, but small and uncorrelated
        track = 0.2 + 0.05 * ((-1.0) ** np.arange(frames.size))
    else:
        track = 0.2 + 1.0 * framp

    run = _make_run(n_oxts=n_oxts, oxts_dt=oxts_dt, frame_every=frame_every,
                    ego_err=ego, track_off=track, gps_window=(LO, HI), mode="gps_dropout")
    base = _make_run(n_oxts=n_oxts, oxts_dt=oxts_dt, frame_every=frame_every,
                     ego_err=0.1, track_off=0.2)
    return run, base


@pytest.mark.parametrize("broken", [None, "a", "b", "c"])
def test_gps_dropout_gate_needs_all_three_conditions(broken):
    run, base = _dropout_pair(broken)
    passed, lines = evaluate("gps_dropout", run, base)
    if broken is None:
        assert passed, lines
        return
    assert not passed, lines
    assert any(f"({broken})" in ln for ln in _fails(lines)), lines


def test_gps_dropout_gate_fails_with_fewer_than_20_usable_frames():
    run, base = _dropout_pair(frame_every=100)       # 1 Hz frames -> 5 inside [4, 8] s
    r, n_usable = _coupling_r(run, (LO, HI))
    assert r is None and n_usable == 5 < COUPLING_MIN_FRAMES

    passed, lines = evaluate("gps_dropout", run, base)
    assert not passed
    failed = _fails(lines)
    assert len(failed) == 1, lines                   # everything else still passes -> not vacuous
    assert "inconclusive" in failed[0] and "5 usable frames" in failed[0]


def test_gps_dropout_correlation_uses_only_in_window_frames():
    n, dt, every = 1200, 0.01, 10
    t = np.arange(n, dtype=float) * dt
    ramp = np.clip((t - LO) / (HI - LO), 0.0, 1.0)
    wobble = np.abs(np.sin(t))                       # out-of-window only, bounded
    in_win = (t >= LO) & (t <= HI)
    ego = np.where(in_win, 0.1 + 1.9 * ramp, 0.1 + 0.9 * wobble)

    frames = t[np.arange(0, n, every)]
    fin = (frames >= LO) & (frames <= HI)
    framp = np.clip((frames - LO) / (HI - LO), 0.0, 1.0)
    # Out of the window the track error moves OPPOSITE to the ego error, and stays inside the
    # 3 m association gate so those frames really are matched and really would be included by
    # a gate that forgot to filter on the window.
    track = np.where(fin, 0.2 + 1.0 * framp, 2.5 - 2.0 * np.abs(np.sin(frames)))

    run = _make_run(n_oxts=n, oxts_dt=dt, frame_every=every, ego_err=ego, track_off=track,
                    gps_window=(LO, HI), mode="gps_dropout")
    _err, cnt = track_error_series(run)
    assert (cnt > 0).all()                           # every frame is matched, in window or not

    r, n_usable = _coupling_r(run, (LO, HI))
    assert n_usable == int(fin.sum()) == 41
    assert r == pytest.approx(1.0, abs=1e-9)         # the out-of-window frames moved nothing


# ---------------------------------------------------------------------------
# imu_bias gate
#
# The magnitude condition measures the INJECTION, which makes it a DIFFERENTIAL against the
# baseline run's final b_x -- not against zero. The ESKF absorbs real KITTI IMU and model error
# into b_x with nothing injected at all (measured +0.0708 m/s^2 on drive_0001), so an absolute
# |b_x| / |injected| test scores 71% on a zero-injection run and is not testing what it claims.
# Every fixture below therefore carries a baseline whose own b_x is NON-ZERO: a baseline pinned
# at 0.0 makes the differential and the absolute numerically identical and leaves this suite
# blind to the difference between them.
# ---------------------------------------------------------------------------

BASELINE_BX = 0.0708      # what the zero-injection baseline converges to on drive_0001


def _bias_run(estimate, injected=0.1, baseline_bx=BASELINE_BX):
    """(run, baseline) for the imu_bias gate, with the baseline's own converged b_x set."""
    run = _make_run(n_oxts=200, imu_bias_xyz=(injected, 0.0, 0.0), accel_bias_x=estimate,
                    mode="imu_bias")
    base = _make_run(n_oxts=200, accel_bias_x=baseline_bx, mode="baseline")
    return run, base


def test_imu_bias_gate_requires_correct_sign():
    # The differential clears 30% by a mile (+0.15 against an injected +0.1), so ONLY the sign
    # check may fail here -- the two conditions have to be independently falsifiable.
    run, base = _bias_run(-0.05, baseline_bx=-0.2)
    passed, lines = evaluate("imu_bias", run, base)
    assert not passed
    failed = _fails(lines)
    assert len(failed) == 1 and "same sign" in failed[0]


@pytest.mark.parametrize("estimate,expected", [
    (BASELINE_BX + 0.025, False),      # differential 25% -- below the floor
    (BASELINE_BX + 0.035, True),       # differential 35% -- clears it
])
def test_imu_bias_gate_requires_30_percent_of_the_differential(estimate, expected):
    """Both cases sit at ~96% and ~106% of the injected magnitude in ABSOLUTE terms, so the old
    |b_x|-based condition passed them both. Only the differential separates them."""
    run, base = _bias_run(estimate)
    passed, lines = evaluate("imu_bias", run, base)
    assert passed is expected, lines


def test_imu_bias_gate_is_not_satisfied_by_the_baselines_own_convergence():
    """A run whose estimate merely reproduces the baseline has learned NOTHING about the
    injection. The old gate scored exactly this at 70.8% and called it a pass."""
    run, base = _bias_run(BASELINE_BX)                  # differential exactly zero
    passed, lines = evaluate("imu_bias", run, base)
    assert not passed
    failed = _fails(lines)
    assert len(failed) == 1 and "differential" in failed[0]
    assert "70.8%" in failed[0]                         # the absolute is still reported


def test_imu_bias_gate_passes_a_large_absolute_when_the_differential_clears():
    """The real 2026-07-28 measurement: b_x = +0.116474 against a baseline of +0.070794 --
    116.5% absolute but 45.7% attributable to the injection. Both numbers must be reported."""
    run, base = _bias_run(0.116474, baseline_bx=0.070794)
    passed, lines = evaluate("imu_bias", run, base)
    assert passed, lines
    assert "45.7%" in lines[-1] and "116.5%" in lines[-1]


@pytest.mark.parametrize("estimate,expected", [(-0.05, True), (0.12, False)])
def test_imu_bias_differential_is_signed_not_absolute(estimate, expected):
    """A NEGATIVE injection wants a NEGATIVE delta. abs(delta)/abs(inj) would score the
    wrong-way case at 49% and applaud a filter that moved the bias the wrong direction."""
    run, base = _bias_run(estimate, injected=-0.1)
    passed, lines = evaluate("imu_bias", run, base)
    assert passed is expected, lines


def test_imu_bias_gate_fails_without_a_baseline():
    """It is a ratio gate now, so a missing baseline is a stated FAIL, never a vacuous pass."""
    run, _base = _bias_run(0.116474)
    passed, lines = evaluate("imu_bias", run, baseline=None)
    assert not passed
    assert len(lines) == 1 and "requires a baseline run" in lines[0]


# ---------------------------------------------------------------------------
# maneuver gate
# ---------------------------------------------------------------------------

MANEUVER_PARAMS = {"maneuver_target": 3, "maneuver_start_s": 5.0, "maneuver_omega": 0.4}


def _maneuver_run(rise_frame, ct=0.8, cv=0.1, ca=0.1, n_frames=120, ct2=0.0):
    # `ct` is the FIRST CT model's probability and `ct2` the second; ct2 defaults to 0.0 so
    # every pre-existing caller keeps the exact bank it had.
    modes = np.zeros((n_frames, N_TARGETS, N_IMM_MODES))
    modes[:, :, 0] = 1.0                              # everyone is CV to start
    modes[rise_frame:, 3] = (cv, ca, ct, ct2)         # target 3's track switches to CT
    return _make_run(n_oxts=n_frames * 10, track_off=0.5, track_mode=modes,
                     mode="maneuver", params=MANEUVER_PARAMS)


@pytest.mark.parametrize("rise_frame,expected", [(50 + 15, True), (50 + 25, False)])
def test_maneuver_gate_requires_ct_rise_within_20_frames(rise_frame, expected):
    passed, lines = evaluate("maneuver", _maneuver_run(rise_frame))
    assert passed is expected, lines
    if not expected:
        assert any("CT probability peaks" in ln for ln in _fails(lines))


def test_maneuver_gate_requires_cv_to_drop():
    # CT clears 0.5 well inside the window, but CV never lets go.
    passed, lines = evaluate("maneuver", _maneuver_run(50 + 10, ct=0.6, cv=0.35, ca=0.05))
    assert not passed
    failed = _fails(lines)
    assert len(failed) == 1 and "CV probability at the CT peak" in failed[0]


def test_maneuver_report_breaks_the_ct_aggregate_out_per_mode():
    """The gate deliberately thresholds the AGGREGATE over every CT model -- "am I turning?" is
    the physical question, and an injected omega of 0.4 rad/s sitting between the configured
    +/-0.25 turn rates legitimately splits its probability across both. That aggregate can
    clear 0.5 with no single CT model near it, so the report must show the split; otherwise the
    writeup cannot say which CT model actually won."""
    run = _maneuver_run(50 + 10, ct=0.3, ct2=0.3, cv=0.2, ca=0.2)
    passed, lines = evaluate("maneuver", run)
    assert passed, lines                              # 0.3 + 0.3 = 0.6 clears the 0.5 aggregate
    peak = next(ln for ln in lines if "CT probability peaks" in ln)
    assert "peaks at 0.6000" in peak                  # the gated quantity is still the sum
    assert "mode probabilities there: CV=0.2000, CA=0.2000, CT0=0.3000, CT1=0.3000" in peak
    assert 0.3 < SPEC_THRESHOLDS["MANEUVER_CT_MIN"]   # neither CT model clears 0.5 alone


# ---------------------------------------------------------------------------
# detection-dropout gates
# ---------------------------------------------------------------------------

DET_LO, DET_HI = 6.0, 7.0


def _gap_run(*, id_change, reacquire_off=0.5, n_frames=120, mode="det_dropout_short"):
    ids = np.tile(10 + np.arange(N_TARGETS, dtype=np.int64), (n_frames, 1))
    frame_t = np.arange(n_frames, dtype=float) * 0.1
    after = frame_t > DET_HI
    if id_change:
        ids[after] += 10                              # the coasting tracks died; new ids are born
    off = np.full((n_frames, N_TARGETS), 0.5)
    first_after = int(np.argmax(after))
    off[first_after] = reacquire_off                  # error at the re-acquisition frame
    return _make_run(n_oxts=n_frames * 10, track_ids=ids, track_off=off,
                     det_window=(DET_LO, DET_HI), mode=mode)


@pytest.mark.parametrize("id_change,expected", [(True, True), (False, False)])
def test_det_dropout_short_gate_requires_an_id_change(id_change, expected):
    run = _gap_run(id_change=id_change, mode="det_dropout_short")
    passed, lines = evaluate("det_dropout_short", run)
    assert passed is expected, lines
    if not expected:
        assert len(_fails(lines)) == N_TARGETS       # every target kept its id: all four fail


@pytest.mark.parametrize("id_change,expected", [(False, True), (True, False)])
def test_det_dropout_coast_gate_requires_id_preservation(id_change, expected):
    run = _gap_run(id_change=id_change, mode="det_dropout_coast")
    passed, lines = evaluate("det_dropout_coast", run)
    assert passed is expected, lines


def test_det_dropout_coast_gate_checks_reacquisition_error():
    # The id survives the gap, but the surviving track comes back 8 m from the target. The
    # re-acquisition check is measured against the SURVIVING track, not through the 3 m
    # association gate -- otherwise an 8 m miss would silently drop out of the comparison.
    run = _gap_run(id_change=False, reacquire_off=8.0, mode="det_dropout_coast")
    passed, lines = evaluate("det_dropout_coast", run)
    assert not passed
    failed = _fails(lines)
    assert len(failed) == N_TARGETS
    assert all("re-acquires" in ln and "8.0000 m" in ln for ln in failed)
    assert COAST_REACQUIRE_MAX_M == 5.0

    ok, _ = evaluate("det_dropout_coast", _gap_run(id_change=False, reacquire_off=0.5,
                                                   mode="det_dropout_coast"))
    assert ok                                         # control: a close re-acquisition passes


def test_across_gap_before_side_stops_strictly_at_the_window_start():
    """`id_before` is the last id seen strictly BEFORE `lo`, never anything from inside the gap.

    Every other fixture in this file only ever changes ids for t > DET_HI, which leaves the
    "before" boundary invisible: `frame_s[k] < lo` and `frame_s[k] <= hi` pick the same id. They
    are not equivalent in general. Here the target is picked up by a different id from mid-gap
    onward -- so the frame at exactly t == DET_HI already carries the POST-gap id. A boundary
    that swept in the gap would latch that as `id_before`, then find the same id still published
    after the gap, and report "no switch" -- failing det_dropout_short on a run that switched.
    """
    n_frames, first, second = 120, 10, 20             # id blocks: 10..13 before, 20..23 after
    frame_t = np.arange(n_frames, dtype=float) * 0.1
    ids = np.tile(first + np.arange(N_TARGETS, dtype=np.int64), (n_frames, 1))
    ids[frame_t >= 6.5] += (second - first)           # the switch happens INSIDE [6.0, 7.0]
    run = _make_run(n_oxts=n_frames * 10, track_ids=ids, track_off=0.5,
                    det_window=(DET_LO, DET_HI), mode="det_dropout_short")

    # Frames at and around t == DET_HI really do carry the post-gap id -- otherwise this fixture
    # would not exercise the boundary at all.
    assert frame_t[70] == DET_HI and ids[70, 0] == second
    assert ids[59, 0] == first                        # ...and the last pre-gap frame carries 10

    spans = _across_gap(run, DET_LO, DET_HI)
    assert {i: (b, a) for i, (b, a, _e, _k) in spans.items()} == \
        {i: (first + i, second + i) for i in range(N_TARGETS)}

    passed, lines = evaluate("det_dropout_short", run)
    assert passed, lines                              # the ids DID change: this must not fail
    assert any("target 0 id 10 -> 20 across the gap" in ln for ln in lines), lines


def test_across_gap_resolves_the_surviving_id_before_the_association_gate():
    """After the gap the SURVIVING pre-gap id is resolved first, by id lookup; the 3 m
    association gate is only the fallback.

    No other fixture separates the two, because none puts a surviving-but-far track and a
    nearer foreign track on the same frame. Here the coasted track comes back 8 m off while a
    foreign id sits 1 m from the target. Resolving through the gate first would hand back the
    foreign id at 1 m -- turning a real 8 m coast failure into a clean-looking re-acquisition
    and leaving the 5 m check vacuous, which is the exact trap `_across_gap` documents.
    """
    n_frames, survivor, foreign = 120, 10, 99
    frame_t = np.arange(n_frames, dtype=float) * 0.1
    after = frame_t > DET_HI
    ids = np.tile(np.array([survivor, foreign], dtype=np.int64), (n_frames, 1))

    xy = np.empty((n_frames, 2, 2))
    xy[:, :, 1] = 0.0
    xy[:, 0, 0] = np.where(after, 28.0, 20.5)         # survivor: 0.5 m off, then 8 m off
    xy[:, 1, 0] = np.where(after, 21.0, FAR_AWAY[0])  # foreign: absent, then 1 m off
    xy[after, 1, 1] = 0.0
    xy[~after, 1, 1] = FAR_AWAY[1]

    run = _make_run(n_oxts=n_frames * 10, target_xy=np.array([[20.0, 0.0]]),
                    track_ids=ids, track_xy=xy, track_count=np.full(n_frames, 2),
                    det_window=(DET_LO, DET_HI), mode="det_dropout_coast")

    k_after = int(np.argmax(after))
    # The fixture really is ambiguous: the association gate on its own prefers the foreign id.
    assert match_tracks(run, k_after) == {0: (foreign, pytest.approx(1.0))}
    assert _across_gap(run, DET_LO, DET_HI) == \
        {0: (survivor, survivor, pytest.approx(8.0), k_after)}

    passed, lines = evaluate("det_dropout_coast", run)
    assert not passed
    assert any(ln.startswith("[PASS]") and f"keeps id {survivor} across the gap "
               f"(saw {survivor} after)" in ln for ln in lines), lines
    failed = _fails(lines)
    assert len(failed) == 1 and "re-acquires" in failed[0] and "8.0000 m" in failed[0]


# ---------------------------------------------------------------------------
# clutter gate
# ---------------------------------------------------------------------------

# 2 and 3 are LITERALS for the same reason as the association-gate boundary above.
@pytest.mark.parametrize("extra,expected", [(2, True), (3, False)])
def test_clutter_gate_allows_baseline_plus_two(extra, expected):
    n_frames = 60
    base = _make_run(n_oxts=n_frames * 10, track_off=0.5)
    n_slots = N_TARGETS + extra
    ids = np.tile(np.concatenate([10 + np.arange(N_TARGETS), 90 + np.arange(extra)]),
                  (n_frames, 1)).astype(np.int64)
    run = _make_run(n_oxts=n_frames * 10, track_off=0.5, track_ids=ids,
                    track_count=np.full(n_frames, n_slots), mode="clutter")
    assert len(confirmed_track_ids(run)) == len(confirmed_track_ids(base)) + extra

    passed, lines = evaluate("clutter", run, base)
    assert passed is expected, lines


# ---------------------------------------------------------------------------
# evaluate() contract
# ---------------------------------------------------------------------------

def test_every_mode_name_is_handled():
    run = _make_run(ego_err=0.2, track_off=0.5)
    base = _make_run(ego_err=0.2, track_off=0.5)
    for mode in MODES:
        passed, lines = evaluate(mode, run, base)
        assert isinstance(passed, bool) and lines, mode
        assert all(ln.startswith(("[PASS]", "[FAIL]")) for ln in lines), mode
    for bogus in ("nonsense", "Baseline", "", "gps-dropout"):
        with pytest.raises(ValueError, match="unknown failure mode"):
            evaluate(bogus, run, base)


def test_gates_evaluate_identically_against_a_real_npz_file(tmp_path):
    """The gates run offline on a saved npz, and `np.load` hands back an NpzFile, not a dict:
    `params_json` arrives as a 0-d '<U' array rather than a `str`, and every column as an array.
    Every other fixture here is an in-memory dict, so nothing else exercises that path."""
    run = _maneuver_run(50 + 15)                      # maneuver is the gate that reads params_json
    path = tmp_path / "pipeline_maneuver.npz"
    np.savez_compressed(path, **run)
    with np.load(path, allow_pickle=False) as npz:
        assert np.asarray(npz["params_json"]).ndim == 0 and not isinstance(npz["params_json"], str)
        for mode in ("maneuver", "baseline"):
            assert evaluate(mode, npz) == evaluate(mode, run), mode   # verdict AND every line


@pytest.mark.parametrize("mode", ["gps_dropout", "clutter", "imu_bias"])
def test_gates_requiring_a_baseline_fail_without_one(mode):
    run = _make_run(ego_err=0.2, track_off=0.5, gps_window=(LO, HI), mode=mode)
    passed, lines = evaluate(mode, run, baseline=None)
    assert not passed
    assert len(lines) == 1 and "requires a baseline run" in lines[0]
