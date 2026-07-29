"""Stage 6 pass/fail gates evaluated over a recorded pipeline run.

Pure numpy -- no ROS import anywhere -- so the gates unit-test on the host against hand-built
arrays and `scripts/plot_stage6.py` can reuse them offline on a saved npz.

A `run` is any mapping carrying the Stage 6 npz schema (an `np.load` NpzFile or a plain dict of
arrays). Every stamp comparison uses the int64-nanosecond columns; the float-second columns are
for plotting and for comparing against the injected dropout windows, never for keying -- float
seconds do not compare equal and would send every alignment down a nearest-match path.

House rule for every gate here: a gate that CANNOT be evaluated -- too few usable frames, a
missing baseline run, no matched track anywhere -- returns False with the reason stated. It
never passes vacuously, because in a run log a vacuous pass is indistinguishable from a
working pipeline.

Thresholds are pre-registered ceilings from the design doc section 6, not values fitted to a
measurement. Ratio gates are computed against the measured baseline run so they adapt to the
drive without being loosened.
"""
from __future__ import annotations

import json
import math

import numpy as np

# --- geometry / layout -----------------------------------------------------------------
# ENU columns. "Horizontal" error is this pair only; height is excluded everywhere.
X_ENU, Y_ENU = 0, 1
# IMM bank order is FIXED as CV, CA, then one CT per configured turn rate
# (kf_tracker/include/kf_tracker/imm.hpp). Anything from index 2 up is a CT mode.
MODE_CV, MODE_CA, FIRST_CT_MODE = 0, 1, 2

MATCH_GATE_M = 3.0
MODES = ("baseline", "gps_dropout", "imu_bias", "maneuver",
         "det_dropout_short", "det_dropout_coast", "clutter")

# --- pre-registered thresholds (design doc section 6) ----------------------------------
BASELINE_EGO_RMSE_MAX = 1.0     # m, pre-registered ceiling
BASELINE_TRACK_RMSE_MAX = 2.0   # m
DROPOUT_EGO_PEAK_RATIO = 3.0
DROPOUT_RECOVERY_RATIO = 2.0
DROPOUT_RECOVERY_S = 2.0
DROPOUT_TRACK_PEAK_RATIO = 2.0
COUPLING_R_MIN = 0.8
COUPLING_MIN_FRAMES = 20
IMU_BIAS_MIN_FRACTION = 0.3
MANEUVER_CT_MIN = 0.5
MANEUVER_CV_MAX = 0.3
MANEUVER_MAX_FRAMES = 20
COAST_REACQUIRE_MAX_M = 5.0
CLUTTER_SLACK = 2

__all__ = [
    "MATCH_GATE_M", "MODES",
    "BASELINE_EGO_RMSE_MAX", "BASELINE_TRACK_RMSE_MAX",
    "DROPOUT_EGO_PEAK_RATIO", "DROPOUT_RECOVERY_RATIO", "DROPOUT_RECOVERY_S",
    "DROPOUT_TRACK_PEAK_RATIO", "COUPLING_R_MIN", "COUPLING_MIN_FRAMES",
    "IMU_BIAS_MIN_FRACTION", "MANEUVER_CT_MIN", "MANEUVER_CV_MAX", "MANEUVER_MAX_FRAMES",
    "COAST_REACQUIRE_MAX_M", "CLUTTER_SLACK",
    "MODE_CV", "MODE_CA", "FIRST_CT_MODE", "X_ENU", "Y_ENU",
    "ego_error", "match_tracks", "track_error_series", "id_switches",
    "confirmed_track_ids", "ego_rmse", "track_rmse", "evaluate",
    "coupling_r", "across_gap",
]


# ---------------------------------------------------------------------------
# Recorded-run accessors
# ---------------------------------------------------------------------------

def _as_str(value) -> str:
    """npz string scalars come back as 0-d unicode arrays; plain dicts carry real str."""
    arr = np.asarray(value)
    return str(arr.item()) if arr.ndim == 0 else str(value)


def _params(run) -> dict:
    """The resolved parameter dict recorded alongside the run. {} if absent or unparsable."""
    try:
        raw = run["params_json"]
    except (KeyError, IndexError):
        return {}
    try:
        parsed = json.loads(_as_str(raw))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _frame_seconds(run) -> np.ndarray:
    return np.asarray(run["frame_t"], dtype=float).reshape(-1)


def _n_frames(run) -> int:
    return int(_frame_seconds(run).size)


def _n_targets(run) -> int:
    return int(np.asarray(run["target_truth_enu"]).shape[1])


def _window(run, key: str) -> tuple[float, float]:
    """Injected dropout window in seconds. [0, 0] (or anything non-positive) means none."""
    w = np.asarray(run[key], dtype=float).reshape(-1)
    if w.size < 2:
        return 0.0, 0.0
    return float(w[0]), float(w[1])


# ---------------------------------------------------------------------------
# Ego error
# ---------------------------------------------------------------------------

def _ego_series(run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(stamps_ns, stamp seconds, horizontal error) for every recorded `/ego/state` sample.

    Alignment to OXTS truth is EXACT nanosecond equality. There is deliberately no
    nearest-stamp fallback: a fallback would silently pair an estimate with the wrong truth
    sample and turn a recording bug into a plausible-looking error curve.
    """
    truth_ns = np.asarray(run["t_ns"]).astype(np.int64).reshape(-1)
    truth_s = np.asarray(run["t"], dtype=float).reshape(-1)
    truth_p = np.asarray(run["ego_truth"], dtype=float).reshape(-1, 3)
    est_ns = np.asarray(run["ego_est_t_ns"]).astype(np.int64).reshape(-1)
    est_p = np.asarray(run["ego_est"], dtype=float).reshape(-1, 3)

    if truth_ns.size != truth_p.shape[0] or truth_ns.size != truth_s.size:
        raise ValueError(f"truth arrays disagree: t_ns={truth_ns.size}, t={truth_s.size}, "
                         f"ego_truth={truth_p.shape[0]}")
    if est_ns.size != est_p.shape[0]:
        raise ValueError(f"ego_est_t_ns={est_ns.size} but ego_est={est_p.shape[0]}")

    lookup = {int(ns): i for i, ns in enumerate(truth_ns.tolist())}
    rows = np.empty(est_ns.size, dtype=np.intp)
    for e, ns in enumerate(est_ns.tolist()):
        row = lookup.get(int(ns))
        if row is None:
            raise ValueError(
                f"/ego/state stamp {ns} ns has no exact match among the {truth_ns.size} "
                f"OXTS truth stamps; alignment is exact-stamp only, never nearest")
        rows[e] = row

    d = est_p[:, [X_ENU, Y_ENU]] - truth_p[rows][:, [X_ENU, Y_ENU]]
    return est_ns, truth_s[rows], np.hypot(d[:, 0], d[:, 1])


def ego_error(run) -> tuple[np.ndarray, np.ndarray]:
    """(stamps_ns, horizontal ego position error) aligned to truth by EXACT stamp.

    Raises ValueError on an ego stamp with no matching truth sample.
    """
    stamps, _secs, err = _ego_series(run)
    return stamps, err


def ego_rmse(run) -> float:
    """RMS horizontal ego error over the whole run. NaN if nothing was recorded."""
    _stamps, err = ego_error(run)
    return float(np.sqrt(np.mean(np.square(err)))) if err.size else float("nan")


# ---------------------------------------------------------------------------
# Track <-> truth matching
# ---------------------------------------------------------------------------

def _match_frame(truth_xy, visible, ids, track_xy, n_tracks) -> dict[int, tuple[int, float]]:
    """Greedy nearest-neighbour in the ENU ground plane, targets taken in index order.

    One track is used at most once, so two targets can never collapse onto the same track.
    The gate is a STRICT `<`: a pair sitting exactly on MATCH_GATE_M is not a match.
    """
    matches: dict[int, tuple[int, float]] = {}
    used: set[int] = set()
    for i in range(truth_xy.shape[0]):
        if not bool(visible[i]):
            continue
        best_j, best_d = -1, math.inf
        for j in range(n_tracks):
            if j in used or int(ids[j]) < 0:
                continue
            px, py = float(track_xy[j, 0]), float(track_xy[j, 1])
            if not (math.isfinite(px) and math.isfinite(py)):
                continue
            d = math.hypot(px - float(truth_xy[i, 0]), py - float(truth_xy[i, 1]))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d < MATCH_GATE_M:
            used.add(best_j)
            matches[i] = (int(ids[best_j]), float(best_d))
    return matches


def match_tracks(run, k: int) -> dict[int, tuple[int, float]]:
    """Frame k: {target_index: (track_id, error_m)}.

    Greedy nearest in the ENU ground plane, targets taken in index order, MATCH_GATE_M cutoff,
    one track used at most once, targets flagged not-visible skipped. Deliberately simpler than
    the Stage 4 motmetrics path -- these are pass/fail signatures, not a benchmark. MOTA/MOTP
    stays a Stage 7 concern.
    """
    truth = np.asarray(run["target_truth_enu"], dtype=float)[k]
    visible = np.asarray(run["target_visible"], dtype=bool)[k]
    ids = np.asarray(run["track_ids"]).astype(np.int64)[k]
    pos = np.asarray(run["track_pos_enu"], dtype=float)[k]
    n_tracks = int(np.asarray(run["track_count"]).reshape(-1)[k])
    n_tracks = max(0, min(n_tracks, int(ids.shape[0])))
    return _match_frame(truth[:, [X_ENU, Y_ENU]], visible, ids,
                        pos[:, [X_ENU, Y_ENU]], n_tracks)


def _all_matches(run) -> list[dict[int, tuple[int, float]]]:
    return [match_tracks(run, k) for k in range(_n_frames(run))]


def track_error_series(run) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (mean matched track error, matched count). NaN where nothing matched."""
    matches = _all_matches(run)
    err = np.full(len(matches), np.nan)
    cnt = np.zeros(len(matches), dtype=int)
    for k, m in enumerate(matches):
        cnt[k] = len(m)
        if m:
            err[k] = float(np.mean([e for _tid, e in m.values()]))
    return err, cnt


def track_rmse(run) -> float:
    """RMS over every individual matched (frame, target) error. NaN if nothing ever matched."""
    errs = [e for m in _all_matches(run) for _tid, e in m.values()]
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def id_switches(run) -> int:
    """Count of target->track_id changes between CONSECUTIVE frames in which the target was
    matched BOTH times.

    A gap where the target is unmatched never fabricates a switch: comparing against the
    last-seen id across a gap would report a switch every time a track is legitimately
    re-born, which is the very thing det_dropout_short measures separately.
    """
    switches = 0
    prev: dict[int, int] = {}
    for m in _all_matches(run):
        cur = {i: tid for i, (tid, _e) in m.items()}
        for i, tid in cur.items():
            if i in prev and prev[i] != tid:
                switches += 1
        prev = cur
    return switches


def confirmed_track_ids(run) -> set[int]:
    """Every distinct track id the tracker published over the run (`/tracks` is confirmed-only)."""
    ids = np.asarray(run["track_ids"]).astype(np.int64)
    counts = np.asarray(run["track_count"]).reshape(-1)
    out: set[int] = set()
    for k in range(ids.shape[0]):
        n = max(0, min(int(counts[k]), int(ids.shape[1])))
        out.update(int(v) for v in ids[k, :n] if int(v) >= 0)
    return out


# ---------------------------------------------------------------------------
# The coupling correlation (design doc section 6, verbatim definition)
# ---------------------------------------------------------------------------

def coupling_r(run, window):
    """Pearson r between in-window ego error and mean matched track error.

    Returns (r, n_usable). r is None when fewer than COUPLING_MIN_FRAMES frames are usable --
    the caller FAILS the gate as inconclusive rather than passing on a two-point correlation.
    """
    lo, hi = float(window[0]), float(window[1])
    stamps, ego_err = ego_error(run)
    ego_by_ns = dict(zip(stamps.tolist(), ego_err.tolist()))
    a, b = [], []
    frame_ns = run["frame_t_ns"]
    frame_s = run["frame_t"]
    for k in range(len(frame_ns)):
        if not (lo <= frame_s[k] <= hi):
            continue
        e = ego_by_ns.get(int(frame_ns[k]))
        if e is None:
            continue                       # no /ego/state at this exact stamp
        m = match_tracks(run, k)
        if not m:
            continue                       # frames with no matched track are excluded
        a.append(e)
        b.append(float(np.mean([err for _, err in m.values()])))
    if len(a) < COUPLING_MIN_FRAMES:
        return None, len(a)                # inconclusive -> the caller FAILS the gate
    return float(np.corrcoef(a, b)[0, 1]), len(a)


_coupling_r = coupling_r        # legacy private name, kept so existing callers/tests still bind


# ---------------------------------------------------------------------------
# Report plumbing
# ---------------------------------------------------------------------------

class _Report:
    """Accumulates one line per checked condition. `ok` is the AND of every condition, and a
    report with no conditions at all is a FAIL -- an empty gate is not a passing gate."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.lines: list[str] = []
        self._oks: list[bool] = []

    def check(self, ok: bool, text: str) -> bool:
        ok = bool(ok)
        self._oks.append(ok)
        self.lines.append(f"[{'PASS' if ok else 'FAIL'}] {self.mode}: {text}")
        return ok

    def fail(self, text: str) -> tuple[bool, list[str]]:
        """Terminal, un-evaluable outcome: record the reason and return a failing result."""
        self.check(False, text)
        return False, self.lines

    def result(self) -> tuple[bool, list[str]]:
        return (bool(self._oks) and all(self._oks)), self.lines


def _needs_baseline(rep: _Report, baseline) -> tuple[bool, list[str]] | None:
    if baseline is None:
        return rep.fail("this is a ratio gate and requires a baseline run; none was supplied")
    return None


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _gate_baseline(mode, run, baseline):
    rep = _Report(mode)
    _stamps, err = ego_error(run)
    if err.size == 0:
        return rep.fail("no /ego/state samples were recorded, so ego RMSE is undefined")
    rmse = float(np.sqrt(np.mean(np.square(err))))
    rep.check(rmse < BASELINE_EGO_RMSE_MAX,
              f"ego_rmse = {rmse:.4f} m (ceiling {BASELINE_EGO_RMSE_MAX:.4f} m, "
              f"{err.size} samples)")

    n_targets = _n_targets(run)
    if _n_frames(run) == 0:
        return rep.fail("no detection frames were recorded")
    seen: set[int] = set()
    for m in _all_matches(run):
        seen.update(m.keys())
    missing = sorted(set(range(n_targets)) - seen)
    rep.check(not missing,
              f"{len(seen)}/{n_targets} targets confirmed at some point"
              + (f"; never confirmed: {missing}" if missing else ""))

    trmse = track_rmse(run)
    if math.isnan(trmse):
        rep.check(False, "no track ever matched a target, so track RMSE is undefined")
    else:
        rep.check(trmse < BASELINE_TRACK_RMSE_MAX,
                  f"track_rmse = {trmse:.4f} m (ceiling {BASELINE_TRACK_RMSE_MAX:.4f} m)")
    return rep.result()


def _gate_gps_dropout(mode, run, baseline):
    rep = _Report(mode)
    early = _needs_baseline(rep, baseline)
    if early is not None:
        return early

    lo, hi = _window(run, "gps_window")
    if not hi > lo:
        return rep.fail(f"no GPS dropout window was injected (gps_window = [{lo}, {hi}])")

    _ns, secs, err = _ego_series(run)
    _bns, _bsecs, berr = _ego_series(baseline)
    if err.size == 0 or berr.size == 0:
        return rep.fail("ego error is empty in the run or the baseline")

    in_win = (secs >= lo) & (secs <= hi)
    if not in_win.any():
        return rep.fail(f"no /ego/state sample falls inside the dropout window [{lo}, {hi}] s")
    peak = float(np.max(err[in_win]))
    base_peak = float(np.max(berr))
    base_rmse = float(np.sqrt(np.mean(np.square(berr))))

    # (a) the dropout must actually hurt: the in-window peak dwarfs the baseline peak.
    rep.check(peak >= DROPOUT_EGO_PEAK_RATIO * base_peak,
              f"(a) in-window ego peak {peak:.4f} m >= {DROPOUT_EGO_PEAK_RATIO:.1f}x baseline "
              f"peak {base_peak:.4f} m (= {DROPOUT_EGO_PEAK_RATIO * base_peak:.4f} m)")

    # (b) and it must recover once GPS returns.
    rec = (secs > hi) & (secs <= hi + DROPOUT_RECOVERY_S)
    if not rec.any():
        rep.check(False, f"(b) no /ego/state sample in the {DROPOUT_RECOVERY_S:.1f} s after "
                         f"GPS returns at t = {hi:.3f} s")
    else:
        best = float(np.min(err[rec]))
        rep.check(best <= DROPOUT_RECOVERY_RATIO * base_rmse,
                  f"(b) ego error falls to {best:.4f} m within {DROPOUT_RECOVERY_S:.1f} s of "
                  f"recovery (limit {DROPOUT_RECOVERY_RATIO:.1f}x baseline RMSE "
                  f"{base_rmse:.4f} m = {DROPOUT_RECOVERY_RATIO * base_rmse:.4f} m)")

    # (c) the coupling: the localization error has to show up in the TRACK positions.
    base_track_rmse = track_rmse(baseline)
    frame_s = _frame_seconds(run)
    terr, _tcnt = track_error_series(run)
    fmask = (frame_s >= lo) & (frame_s <= hi) & np.isfinite(terr)
    if math.isnan(base_track_rmse):
        rep.check(False, "(c) the baseline run has no matched track, so its track RMSE is "
                         "undefined")
    elif not fmask.any():
        rep.check(False, "(c) no detection frame inside the window has a matched track")
    else:
        tpeak = float(np.max(terr[fmask]))
        rep.check(tpeak >= DROPOUT_TRACK_PEAK_RATIO * base_track_rmse,
                  f"(c) in-window track error peak {tpeak:.4f} m >= "
                  f"{DROPOUT_TRACK_PEAK_RATIO:.1f}x baseline track RMSE "
                  f"{base_track_rmse:.4f} m "
                  f"(= {DROPOUT_TRACK_PEAK_RATIO * base_track_rmse:.4f} m)")

    r, n_usable = _coupling_r(run, (lo, hi))
    if r is None:
        rep.check(False, f"(c) coupling correlation is inconclusive: {n_usable} usable frames, "
                         f"{COUPLING_MIN_FRAMES} required")
    else:
        rep.check(r > COUPLING_R_MIN,
                  f"(c) coupling r = {r:.4f} > {COUPLING_R_MIN:.2f} over {n_usable} frames")
    return rep.result()


def _gate_imu_bias(mode, run, baseline):
    """Convergence of the ESKF accel-bias estimate onto an INJECTED body-x bias.

    This is a RATIO gate, and the ratio is against the baseline run's final b_x -- not against
    zero. The ESKF absorbs real KITTI IMU and model error into b_x whether or not anything is
    injected: on drive_0001 the zero-injection baseline already converges to b_x = +0.0708 m/s^2,
    so an absolute |b_x| / |injected| test scores 71% on a run with NO injection at all. It would
    be measuring the drive, not the injection. The gated quantity is therefore the
    injection-attributable differential (b_x_run - b_x_baseline) / injected; the absolute figure
    is still reported beside it, because the writeup needs both.
    """
    rep = _Report(mode)
    early = _needs_baseline(rep, baseline)
    if early is not None:
        return early

    injected = np.asarray(run["imu_bias_xyz"], dtype=float).reshape(-1)
    if injected.size < 1 or injected[0] == 0.0:
        return rep.fail("no accelerometer bias was injected on body x, so convergence cannot "
                        "be judged")
    inj = float(injected[0])
    est = np.asarray(run["ego_accel_bias"], dtype=float).reshape(-1, 3)
    if est.shape[0] == 0:
        return rep.fail("no ESKF bias estimates were recorded")
    bx = float(est[-1, X_ENU])

    base_est = np.asarray(baseline["ego_accel_bias"], dtype=float).reshape(-1, 3)
    if base_est.shape[0] == 0:
        return rep.fail("the baseline run recorded no ESKF bias estimates, so the "
                        "injection-attributable differential cannot be formed")
    bx_base = float(base_est[-1, X_ENU])

    rep.check(bx * inj > 0.0,
              f"final accel-bias estimate b_x = {bx:+.5f} m/s^2 has the same sign as the "
              f"injected {inj:+.5f} m/s^2")

    delta = bx - bx_base
    frac = delta / inj                       # signed: a negative injection wants a negative delta
    abs_frac = abs(bx) / abs(inj)
    rep.check(frac >= IMU_BIAS_MIN_FRACTION,
              f"differential b_x = {bx:+.5f} - {bx_base:+.5f} (baseline, zero injection) = "
              f"{delta:+.5f} m/s^2 is {frac * 100.0:.1f}% of the injected {inj:+.5f} m/s^2 "
              f"(minimum {IMU_BIAS_MIN_FRACTION * 100.0:.0f}%); the absolute |b_x| is "
              f"{abs_frac * 100.0:.1f}% and is NOT the gated quantity")
    return rep.result()


def _row_of_track_id(run, k: int, track_id: int) -> int:
    ids = np.asarray(run["track_ids"]).astype(np.int64)[k]
    n = max(0, min(int(np.asarray(run["track_count"]).reshape(-1)[k]), int(ids.shape[0])))
    for j in range(n):
        if int(ids[j]) == track_id:
            return j
    return -1


def _gate_maneuver(mode, run, baseline):
    rep = _Report(mode)
    params = _params(run)
    target = int(params.get("maneuver_target", -1))
    if not 0 <= target < _n_targets(run):
        return rep.fail(f"params_json has no valid maneuver_target (got {target!r}); the run "
                        f"cannot be checked for a mode switch")
    start_s = float(params.get("maneuver_start_s", float("nan")))
    if not math.isfinite(start_s):
        return rep.fail("params_json has no maneuver_start_s")

    frame_s = _frame_seconds(run)
    if frame_s.size == 0:
        return rep.fail("no detection frames were recorded")
    onset_t = float(frame_s[0]) + start_s     # same rule targets.py uses: relative to frame 0
    onset = int(np.argmax(frame_s >= onset_t)) if bool((frame_s >= onset_t).any()) else -1
    if onset < 0:
        return rep.fail(f"the maneuver onset t = {onset_t:.3f} s is after the last frame "
                        f"({frame_s[-1]:.3f} s)")

    probs = np.asarray(run["track_mode"], dtype=float)
    if probs.ndim != 3 or probs.shape[2] <= FIRST_CT_MODE:
        return rep.fail(f"the recorded IMM bank has {0 if probs.ndim != 3 else probs.shape[2]} "
                        f"modes, so it carries no CT mode to check")

    last = min(onset + MANEUVER_MAX_FRAMES, len(frame_s) - 1)
    samples: list[tuple[int, float, float]] = []
    for k in range(onset, last + 1):
        m = match_tracks(run, k)
        if target not in m:
            continue
        j = _row_of_track_id(run, k, m[target][0])
        if j < 0:
            continue
        mu = probs[k, j]
        if not np.all(np.isfinite(mu)):
            continue
        samples.append((k, float(np.sum(mu[FIRST_CT_MODE:])), float(mu[MODE_CV]), mu))
    if not samples:
        return rep.fail(f"target {target} was never matched to a track in the {MANEUVER_MAX_FRAMES}"
                        f" frames after the onset (frames {onset}..{last})")

    k_peak, ct_peak, cv_at_peak, mu_peak = max(samples, key=lambda s: s[1])
    # The GATED quantity is the aggregate over every CT model -- "am I turning?" is the
    # physical question, and an injected omega between two configured turn rates legitimately
    # splits its probability across both. The per-mode breakdown is reported (never gated) so
    # the writeup can say which single CT model actually won at the peak.
    breakdown = ", ".join(
        [f"CV={float(mu_peak[MODE_CV]):.4f}", f"CA={float(mu_peak[MODE_CA]):.4f}"]
        + [f"CT{n}={float(p):.4f}" for n, p in enumerate(mu_peak[FIRST_CT_MODE:])])
    rep.check(ct_peak > MANEUVER_CT_MIN,
              f"target {target} CT probability peaks at {ct_peak:.4f} "
              f"(> {MANEUVER_CT_MIN:.2f}) at frame {k_peak}, {k_peak - onset} frames after the "
              f"onset (limit {MANEUVER_MAX_FRAMES}); mode probabilities there: {breakdown}")
    rep.check(cv_at_peak < MANEUVER_CV_MAX,
              f"target {target} CV probability at the CT peak is {cv_at_peak:.4f} "
              f"(< {MANEUVER_CV_MAX:.2f})")
    return rep.result()


def across_gap(run, lo: float, hi: float):
    """{target: (id_before, id_after, reacquire_error_m, reacquire_frame)} for every target with
    evidence on BOTH sides of the detection-dropout window.

    `id_before` is the last id matched to the target strictly before the gap. The after side is
    the FIRST visible frame past the gap carrying evidence, resolved in this order:

      1. the pre-gap track id is still being published -- it coasted through the gap. Its error
         is measured directly against the target truth, deliberately NOT through MATCH_GATE_M:
         a track that survived but came back 8 m off must show up as a large re-acquisition
         error, not silently drop out of the comparison and leave the 5 m check vacuous.
      2. otherwise, whatever the association gate matches -- the pre-gap track is gone, so the
         target has been picked up by a newly born id, which is the ID switch we are looking for.
    """
    frame_s = _frame_seconds(run)
    matches = _all_matches(run)
    truth = np.asarray(run["target_truth_enu"], dtype=float)
    visible = np.asarray(run["target_visible"], dtype=bool)
    ids = np.asarray(run["track_ids"]).astype(np.int64)
    pos = np.asarray(run["track_pos_enu"], dtype=float)
    counts = np.asarray(run["track_count"]).reshape(-1)

    before: dict[int, int] = {}
    for k in range(frame_s.size):
        if frame_s[k] < lo:
            for i, (tid, _e) in matches[k].items():
                before[i] = tid

    out: dict[int, tuple[int, int, float, int]] = {}
    for i, id_before in sorted(before.items()):
        for k in range(frame_s.size):
            if frame_s[k] <= hi or not bool(visible[k, i]):
                continue
            n = max(0, min(int(counts[k]), int(ids.shape[1])))
            row = next((j for j in range(n) if int(ids[k, j]) == id_before), -1)
            if row >= 0 and np.all(np.isfinite(pos[k, row, [X_ENU, Y_ENU]])):
                d = math.hypot(float(pos[k, row, X_ENU]) - float(truth[k, i, X_ENU]),
                               float(pos[k, row, Y_ENU]) - float(truth[k, i, Y_ENU]))
                out[i] = (id_before, id_before, float(d), k)
                break
            if i in matches[k]:
                tid, e = matches[k][i]
                out[i] = (id_before, int(tid), float(e), k)
                break
    return out


_across_gap = across_gap        # legacy private name, kept so existing callers/tests still bind


def _gate_det_dropout_short(mode, run, baseline):
    rep = _Report(mode)
    lo, hi = _window(run, "det_window")
    if not hi > lo:
        return rep.fail(f"no detection dropout window was injected (det_window = [{lo}, {hi}])")
    spans = _across_gap(run, lo, hi)
    if not spans:
        return rep.fail(f"no target is matched on both sides of the [{lo}, {hi}] s gap, so "
                        f"'did the id change?' cannot be answered")
    for i, (id_before, id_after, _err, _k) in spans.items():
        rep.check(id_before != id_after,
                  f"target {i} id {id_before} -> {id_after} across the gap "
                  f"(max_age too small to coast: an ID switch is the expected outcome)")
    return rep.result()


def _gate_det_dropout_coast(mode, run, baseline):
    rep = _Report(mode)
    lo, hi = _window(run, "det_window")
    if not hi > lo:
        return rep.fail(f"no detection dropout window was injected (det_window = [{lo}, {hi}])")
    spans = _across_gap(run, lo, hi)
    if not spans:
        return rep.fail(f"no target is matched on both sides of the [{lo}, {hi}] s gap, so "
                        f"survival cannot be shown")
    for i, (id_before, id_after, err, k) in spans.items():
        rep.check(id_before == id_after,
                  f"target {i} keeps id {id_before} across the gap (saw {id_after} after)")
        rep.check(err < COAST_REACQUIRE_MAX_M,
                  f"target {i} re-acquires at frame {k} with error {err:.4f} m "
                  f"(< {COAST_REACQUIRE_MAX_M:.1f} m)")
    return rep.result()


def _gate_clutter(mode, run, baseline):
    rep = _Report(mode)
    early = _needs_baseline(rep, baseline)
    if early is not None:
        return early

    n_tracks = len(confirmed_track_ids(run))
    n_base = len(confirmed_track_ids(baseline))
    rep.check(n_tracks <= n_base + CLUTTER_SLACK,
              f"{n_tracks} confirmed tracks vs {n_base} in the baseline "
              f"(limit {n_base + CLUTTER_SLACK})")

    sw = id_switches(run)
    sw_base = id_switches(baseline)
    rep.check(sw <= sw_base + CLUTTER_SLACK,
              f"{sw} ID switches vs {sw_base} in the baseline (limit {sw_base + CLUTTER_SLACK})")
    return rep.result()


_GATES = {
    "baseline": _gate_baseline,
    "gps_dropout": _gate_gps_dropout,
    "imu_bias": _gate_imu_bias,
    "maneuver": _gate_maneuver,
    "det_dropout_short": _gate_det_dropout_short,
    "det_dropout_coast": _gate_det_dropout_coast,
    "clutter": _gate_clutter,
}
assert tuple(_GATES) == MODES, "every MODES entry needs a gate and vice versa"


def evaluate(mode: str, run, baseline=None) -> tuple[bool, list[str]]:
    """Evaluate one mode's gate. Returns (passed, report lines).

    An unknown mode raises ValueError -- a typo must be fatal, never a silently-skipped gate.
    A ratio gate called with baseline=None returns (False, [... requires a baseline run]).
    """
    if mode not in _GATES:
        raise ValueError(f"unknown Stage 6 mode {mode!r}; expected one of {list(MODES)}")
    passed, lines = _GATES[mode](mode, run, baseline)
    return bool(passed), list(lines)
