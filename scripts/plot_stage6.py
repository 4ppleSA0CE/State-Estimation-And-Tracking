"""Render the six Stage 6 figures from the recorded per-mode runs.

Runs on the HOST. matplotlib is not in the ROS2 image, so the container records the npz files and
the host draws them — the same split scripts/plot_tracker_parity.py documents:

    # container:
    python3 /workspace/scripts/run_stage6.py            -> data/cache/stage6_<mode>.npz
    # host, from the repo root:
    python3 scripts/plot_stage6.py                      -> docs/images/stage6_<figure>.png

Every metric on every figure comes from kf_bringup.stage6_gates — the same module pipeline_replay
calls for its verdict — so a figure and its gate can never disagree. Each figure also carries the
gate's own report lines as a footer, verbatim and unparsed, so the picture and the PASS/FAIL are
read together.

Time axes are the float-second columns the npz already carries (`t`, `frame_t`), because those are
the base the injected dropout windows are recorded in and the base the gates compare against. The
int64 nanosecond columns are the keys; they are never used as an axis.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402  after the backend is pinned

ROOT = Path(__file__).resolve().parent.parent
# APPEND, never insert, for the same reason scripts/run_stage6.py does: where a kf_bringup is
# already installed, that install is the one the pipeline ran, so it must win. On the host --
# which is where this script belongs, matplotlib not being in the image -- nothing is
# installed and the source tree below is what resolves.
sys.path.append(str(ROOT / "ros2_ws" / "src" / "kf_bringup"))

from kf_bringup import stage6_gates as gates  # noqa: E402  metrics shared with the live gate

DEFAULT_CACHE = ROOT / "data" / "cache"
DEFAULT_IMAGES = ROOT / "docs" / "images"
MAX_FOOTER_LINES = 12
SHADE = {"color": "0.80", "alpha": 0.55, "zorder": 0, "lw": 0}


# --------------------------------------------------------------------------------------------
# loading and small shared helpers
# --------------------------------------------------------------------------------------------
def load_run(mode: str, cache: Path) -> dict | None:
    """The recorded run as a plain dict. None if that mode was never run."""
    path = cache / f"stage6_{mode}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def frames_s(run: dict) -> np.ndarray:
    """Detection-frame times, in the run's own float-second base."""
    return np.asarray(run["frame_t"], dtype=float).reshape(-1)


def ego_err(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """(frame-time-compatible seconds, horizontal ego error), aligned by the gate itself.

    stage6_gates.ego_error has already proved every `/ego/state` stamp has an exact truth match
    (it raises otherwise), so the ns -> float-second lookup below cannot miss.
    """
    stamps, err = gates.ego_error(run)
    t_ns = np.asarray(run["t_ns"], dtype=np.int64).reshape(-1)
    t_s = np.asarray(run["t"], dtype=float).reshape(-1)
    lookup = {int(ns): i for i, ns in enumerate(t_ns.tolist())}
    rows = [lookup[int(ns)] for ns in np.asarray(stamps, dtype=np.int64).tolist()]
    return t_s[rows], np.asarray(err, dtype=float)


def track_err(run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(seconds, per-frame mean matched track error, matched count)."""
    err, count = gates.track_error_series(run)
    return frames_s(run), np.asarray(err, dtype=float), np.asarray(count, dtype=float)


def _window(run: dict, key: str) -> tuple[float, float] | None:
    """The injected [start, end] s window, or None for the [0, 0] "not injected" sentinel."""
    lo, hi = np.asarray(run[key], dtype=float).reshape(-1)[:2]
    return (float(lo), float(hi)) if hi > lo else None


def _shade(ax, window, label=None, **kwargs) -> None:
    if window is not None:
        ax.axvspan(window[0], window[1], label=label, **{**SHADE, **kwargs})


def _param(run: dict, name: str, default):
    """One resolved node parameter out of `params_json`, read EXACTLY as the gate reads it.

    Delegating to `gates._params` is the whole point: an independent reader here (one that
    walked nested dicts, say) makes the figure and the gate disagree about the same run. The
    2026-07-28 review produced a fully-populated panel titled "Maneuver — track 3 on target 3"
    sitting above a footer reading "params_json has no valid maneuver_target (got -1)".
    `pipeline_replay._params_dict` writes the dict FLAT so that one reader suffices.
    """
    return gates._params(run).get(name, default)


def _max_age(run: dict, assumed: int) -> str:
    """`max_age=<n>` for a legend, read from the run; `<n> (assumed)` when it was not recorded.

    Never silently prints the preset value as though it had been measured: the whole point of
    the det_dropout pair is that max_age is the only thing that differs between them, so a
    label that cannot be traced to the run has to say so.
    """
    value = _param(run, "max_age", None)
    return f"max_age={value}" if value is not None else f"max_age={assumed} (assumed)"


def _nanmax(values) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def _n_tracks(run: dict, k: int) -> int:
    """Usable rows of the track arrays at frame k, clamped exactly as `gates.match_tracks` is.

    `track_count` is what the tracker published; the recorded arrays are only K wide. A frame
    that overflowed K is truncated in the npz, so iterating the raw count would IndexError here
    while the gate -- which clamps -- scored the run happily. Figure and gate must see the same
    rows, so the clamp is copied rather than assumed away.
    """
    n = int(np.asarray(run["track_count"]).reshape(-1)[k])
    return max(0, min(n, int(np.asarray(run["track_ids"]).shape[1])))


def _track_paths(run: dict) -> dict[int, np.ndarray]:
    """{track_id: (n, 3) array of (t_s, east, north)} from the de-permuted ENU positions."""
    ids, pos = run["track_ids"], run["track_pos_enu"]
    t = frames_s(run)
    paths: dict[int, list[tuple[float, float, float]]] = {}
    for k in range(int(t.size)):
        for j in range(_n_tracks(run, k)):
            tid = int(ids[k, j])
            p = np.asarray(pos[k, j], dtype=float)
            if tid >= 0 and np.isfinite(p[:2]).all():
                paths.setdefault(tid, []).append((float(t[k]), float(p[0]), float(p[1])))
    return {tid: np.asarray(rows) for tid, rows in paths.items()}


def _matched_ids(run: dict) -> dict[int, np.ndarray]:
    """{target index: (n, 2) array of (t_s, track id)} using the gate's own matcher."""
    t = frames_s(run)
    out: dict[int, list[tuple[float, int]]] = {}
    for k in range(int(t.size)):
        for target, (tid, _err) in gates.match_tracks(run, k).items():
            out.setdefault(int(target), []).append((float(t[k]), int(tid)))
    return {target: np.asarray(rows) for target, rows in out.items()}


def _track_for_target(run: dict, target: int) -> int | None:
    """The track id most often matched to `target` across the run."""
    counts: dict[int, int] = {}
    for k in range(int(frames_s(run).size)):
        hit = gates.match_tracks(run, k).get(target)
        if hit is not None:
            counts[int(hit[0])] = counts.get(int(hit[0]), 0) + 1
    return max(counts, key=counts.__getitem__) if counts else None


def _mode_labels(n_modes: int) -> list[str]:
    """IMM bank order, fixed by imm.hpp: CV, CA, then one filter per configured CT turn rate.

    The CT index is 0-based to match, character for character, the per-mode breakdown the gate
    prints into the footer of this same figure ("CV=..., CA=..., CT0=..., CT1=..."). A 1-based
    legend over a 0-based footer names the same curve two different things on one page.
    """
    if n_modes <= gates.FIRST_CT_MODE:
        return [f"mode {i}" for i in range(n_modes)]
    labels = [""] * n_modes
    labels[gates.MODE_CV] = "CV"
    labels[gates.MODE_CA] = "CA"
    for i in range(gates.FIRST_CT_MODE, n_modes):
        labels[i] = f"CT{i - gates.FIRST_CT_MODE}"
    return labels


def _mode_series(run: dict, tid: int) -> tuple[np.ndarray, np.ndarray]:
    """(t_s, (n, B) mode probabilities) for one track id."""
    ids, probs = run["track_ids"], run["track_mode"]
    t = frames_s(run)
    ts: list[float] = []
    rows: list[np.ndarray] = []
    for k in range(int(t.size)):
        for j in range(_n_tracks(run, k)):
            row = np.asarray(probs[k, j], dtype=float)
            if int(ids[k, j]) == tid and np.isfinite(row).all():
                ts.append(float(t[k]))
                rows.append(row)
                break
    if not rows:
        return np.zeros(0), np.zeros((0, 0))
    return np.asarray(ts), np.asarray(rows)


def _coupling_points(run: dict, window) -> tuple[np.ndarray, np.ndarray]:
    """The (ego error, mean matched track error) pairs the coupling gate correlates.

    Mirrors stage6_gates.coupling_r's frame filter exactly — same window test against the float
    `frame_t` column, same exact-stamp ego lookup, same "no matched track excludes the frame" —
    so the scatter holds the frames the gate scored. `r` itself comes from the gate, never here.
    """
    lo, hi = window
    stamps, err = gates.ego_error(run)
    by_ns = dict(zip(np.asarray(stamps, dtype=np.int64).tolist(),
                     np.asarray(err, dtype=float).tolist()))
    frame_ns = np.asarray(run["frame_t_ns"], dtype=np.int64).reshape(-1)
    frame_s = frames_s(run)
    a: list[float] = []
    b: list[float] = []
    for k in range(frame_ns.size):
        if not (lo <= frame_s[k] <= hi):
            continue
        e = by_ns.get(int(frame_ns[k]))
        if e is None:
            continue
        matched = gates.match_tracks(run, k)
        if not matched:
            continue
        a.append(e)
        b.append(float(np.mean([m_err for _tid, m_err in matched.values()])))
    return np.asarray(a), np.asarray(b)


# --------------------------------------------------------------------------------------------
# figure assembly
# --------------------------------------------------------------------------------------------
def _gate_text(mode: str, run: dict, baseline: dict | None = None) -> str:
    """The gate's verdict for this run, verbatim. A raising gate is reported, never swallowed."""
    try:
        passed, lines = gates.evaluate(mode, run, baseline)
        head = f"{mode}: {'PASS' if passed else 'FAIL'}"
    except Exception as exc:                                   # noqa: BLE001 — diagnostic figure
        print(f"WARNING: stage6_gates.evaluate({mode!r}) raised "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        head, lines = f"{mode}: GATE ERROR", [f"{type(exc).__name__}: {exc}"]
    lines = list(lines)
    shown = lines[:MAX_FOOTER_LINES]
    if len(lines) > MAX_FOOTER_LINES:
        shown.append(f"... {len(lines) - MAX_FOOTER_LINES} more line(s) omitted")
    return "\n".join([head] + [f"    {s}" for s in shown])


def _wrap_footer(gate_texts: list[str], fig, fontsize: float) -> list[str]:
    """The gate lines hard-wrapped to the canvas width, so none runs off the right edge.

    The footer is the gate's report verbatim, and a line that leaves the canvas is not verbatim
    -- it is silently truncated. `maneuver`'s per-mode breakdown line is ~200 characters by
    design (design doc 6.1 requires it), which is twice what a 10 in canvas holds at 7 pt.
    DejaVu Sans Mono advances 0.6 em per glyph; 2 % of the width is kept as a right margin.
    """
    cols = max(40, int(fig.get_figwidth() * 72.0 * 0.98 / (0.6 * fontsize)))
    out: list[str] = []
    for line in "\n".join(gate_texts).splitlines():
        out.extend(textwrap.wrap(line, cols, subsequent_indent=" " * 8,
                                 break_long_words=True, break_on_hyphens=False) or [""])
    return out


def _save(fig, name: str, gate_texts: list[str], images: Path) -> Path:
    fontsize = 7.0
    lines = _wrap_footer(gate_texts, fig, fontsize)
    if len(lines) > 10:
        fontsize = 6.0
        lines = _wrap_footer(gate_texts, fig, fontsize)
    row_h = fontsize * 1.4 / 72.0
    # Never let the footer eat more than 30 % of the figure; drop the overflow with a notice
    # rather than letting it run under the axes, where it would be unreadable anyway.
    max_rows = max(1, int(0.30 * fig.get_figheight() / row_h))
    if len(lines) > max_rows:
        lines = lines[:max_rows - 1] + [f"... {len(lines) - max_rows + 1} more footer row(s) "
                                        f"omitted"]
    text = "\n".join(lines)
    reserved = min(0.30, (len(lines) * row_h) / fig.get_figheight() + 0.012)
    fig.tight_layout(rect=(0.0, reserved, 1.0, 1.0))
    fig.text(0.01, 0.006, text, fontsize=fontsize, family="monospace", va="bottom", ha="left")
    images.mkdir(parents=True, exist_ok=True)
    out = images / f"stage6_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def fig_baseline(runs: dict, images: Path) -> Path:
    run = runs["baseline"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                  gridspec_kw={"height_ratios": [3, 1]})

    truth = np.asarray(run["target_truth_enu"], dtype=float)
    for i in range(truth.shape[1]):
        ax.plot(truth[:, i, 0], truth[:, i, 1], "--", color="0.60", lw=1.6, zorder=1)
    for tid, path in sorted(_track_paths(run).items()):
        ax.plot(path[:, 1], path[:, 2], "-", color=f"C{tid % 10}", lw=1.1,
                marker="o", ms=2.5, zorder=3)
    ego_t = np.asarray(run["ego_truth"], dtype=float)
    ego_e = np.asarray(run["ego_est"], dtype=float)
    ax.plot(ego_t[:, 0], ego_t[:, 1], "-", color="k", lw=2.6, zorder=2)
    # The ego estimate must NOT use a "C<n>" colour: the tracks are drawn on the C0-C9 cycle,
    # so "C3" would be pixel-identical to track id 3 while the legend called red the ego
    # estimate. Black dashed sits outside the track cycle; the white underlay keeps the dashes
    # legible where the estimate lies on top of the (black) truth line, which is most of it.
    ax.plot(ego_e[:, 0], ego_e[:, 1], "-", color="w", lw=2.4, zorder=4)
    ax.plot(ego_e[:, 0], ego_e[:, 1], "--", color="k", lw=1.4, dashes=(4, 3), zorder=5)

    ax.plot([], [], "--", color="0.60", lw=1.6, label="target truth (ENU)")
    ax.plot([], [], "-", color="C0", marker="o", ms=3, label="tracks, de-permuted to ENU")
    ax.plot([], [], "-", color="k", lw=2.6, label="ego truth (OXTS)")
    ax.plot([], [], "--", color="k", lw=1.4, dashes=(4, 3), label="ego estimate (ESKF)")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(f"Stage 6 baseline — ego RMSE {gates.ego_rmse(run):.3f} m "
                 f"(ceiling {gates.BASELINE_EGO_RMSE_MAX:g}), "
                 f"track RMSE {gates.track_rmse(run):.3f} m "
                 f"(ceiling {gates.BASELINE_TRACK_RMSE_MAX:g})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    t_f, terr, tcnt = track_err(run)
    ax2.plot(t_f, terr, "-", color="C0", lw=1.3, label="mean matched track error")
    ax2.axhline(gates.BASELINE_TRACK_RMSE_MAX, ls="--", color="k", lw=1,
                label=f"ceiling {gates.BASELINE_TRACK_RMSE_MAX:g} m")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("track error [m]")
    ax2.grid(True, alpha=0.3)
    ax3 = ax2.twinx()
    ax3.step(t_f, tcnt, where="mid", color="0.55", lw=1.0, label="matched targets")
    ax3.set_ylabel("matched targets")
    ax3.set_ylim(-0.2, (_nanmax(tcnt) if tcnt.size else 1.0) + 0.5)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    return _save(fig, "baseline", [_gate_text("baseline", run)], images)


def fig_gps_dropout(runs: dict, images: Path) -> Path:
    run, base = runs["gps_dropout"], runs["baseline"]
    window = _window(run, "gps_window")
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    tb, eb = ego_err(base)
    td, ed = ego_err(run)
    _shade(ax, window, label="GPS suppressed")
    if window is not None:
        # condition (b) is scored in this trailing band, so draw it as its own region
        ax.axvspan(window[1], window[1] + gates.DROPOUT_RECOVERY_S, color="C2", alpha=0.13,
                   lw=0, zorder=0, label=f"{gates.DROPOUT_RECOVERY_S:g} s recovery window")
    ax.plot(tb, eb, "-", color="0.55", lw=1.2, label="baseline")
    ax.plot(td, ed, "-", color="C3", lw=1.4, label="gps_dropout")
    base_rmse = gates.ego_rmse(base)
    ax.axhline(gates.DROPOUT_RECOVERY_RATIO * base_rmse, ls=":", color="C2", lw=1.2,
               label=f"(b) {gates.DROPOUT_RECOVERY_RATIO:g}× baseline ego RMSE")
    in_win = np.ones_like(td, dtype=bool) if window is None else \
        (td >= window[0]) & (td <= window[1])
    peak_d, peak_b = _nanmax(ed[in_win]), _nanmax(eb)
    ratio = peak_d / peak_b if peak_b else float("nan")
    ax.set_ylabel("ego position error [m]")
    ax.set_title(f"GPS dropout — (a) in-window ego peak {peak_d:.2f} m vs baseline peak "
                 f"{peak_b:.2f} m = {ratio:.1f}× (gate ≥ {gates.DROPOUT_EGO_PEAK_RATIO:g}×)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    tfb, errb, _ = track_err(base)
    tfd, errd, _ = track_err(run)
    base_track_rmse = gates.track_rmse(base)
    _shade(ax2, window)
    ax2.plot(tfb, errb, "-", color="0.55", lw=1.2, label="baseline")
    ax2.plot(tfd, errd, "-", color="C0", lw=1.4, label="gps_dropout")
    ax2.axhline(gates.DROPOUT_TRACK_PEAK_RATIO * base_track_rmse, ls="--", color="k", lw=1,
                label=f"(c) {gates.DROPOUT_TRACK_PEAK_RATIO:g}× baseline track RMSE "
                      f"({base_track_rmse:.2f} m)")
    in_frames = np.ones_like(tfd, dtype=bool) if window is None else \
        (tfd >= window[0]) & (tfd <= window[1])
    ax2.set_ylabel("mean matched track error [m]")
    ax2.set_xlabel("t [s]")
    ax2.set_title(f"(c) in-window track error peak "
                  f"{_nanmax(errd[in_frames]):.2f} m — the localization error reaching the tracks")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    if window is None:
        ax3.set_title("no GPS dropout window recorded — nothing to correlate")
    else:
        a, b = _coupling_points(run, window)
        r, n_used = gates.coupling_r(run, window)
        ax3.scatter(a, b, s=18, c="C3", zorder=3)
        if a.size >= 2 and np.ptp(a) > 0:
            slope, intercept = np.polyfit(a, b, 1)
            xs = np.linspace(a.min(), a.max(), 2)
            ax3.plot(xs, slope * xs + intercept, "-", color="k", lw=1.2,
                     label=f"least squares, {slope:.2f} m track error per m of ego error")
            ax3.legend(loc="best", fontsize=8)
        shown = "n/a (inconclusive)" if r is None else f"{r:.3f}"
        ax3.set_title(f"(c) coupling inside the window — r = {shown} "
                      f"(gate > {gates.COUPLING_R_MIN:g}), {n_used} usable frames "
                      f"(gate ≥ {gates.COUPLING_MIN_FRAMES})")
    ax3.set_xlabel("ego position error [m]")
    ax3.set_ylabel("mean matched track error [m]")
    ax3.grid(True, alpha=0.3)

    return _save(fig, "gps_dropout", [_gate_text("gps_dropout", run, base)], images)


def fig_imu_bias(runs: dict, images: Path) -> Path:
    run, base = runs["imu_bias"], runs["baseline"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    t, _ = ego_err(run)
    bx = np.asarray(run["ego_accel_bias"], dtype=float).reshape(-1, 3)[:, 0]
    base_bx = np.asarray(base["ego_accel_bias"], dtype=float).reshape(-1, 3)[:, 0]
    injected = float(np.asarray(run["imu_bias_xyz"], dtype=float).reshape(-1)[0])
    final = float(bx[-1]) if bx.size else float("nan")
    # The gated quantity is the DIFFERENTIAL against the baseline's own converged b_x, because
    # the ESKF absorbs real IMU/model error into b_x with nothing injected at all. The absolute
    # is drawn too, but the gate line and the headline number must be the differential or the
    # figure asserts a gate that no longer exists.
    base_final = float(base_bx[-1]) if base_bx.size else float("nan")
    fraction = abs(final) / abs(injected) if injected else float("nan")
    diff_fraction = (final - base_final) / injected if injected else float("nan")

    tb, _ = ego_err(base)
    ax.plot(tb, base_bx, "-", color="0.60", lw=1.0,
            label=f"baseline, no injection (final {base_final:+.4f})")
    ax.plot(t[:bx.size], bx, "-", color="C0", lw=1.5, label="imu_bias run, estimated $b_{a,x}$")
    ax.axhline(injected, ls="--", color="k", lw=1.2, label=f"injected {injected:+.3f} m/s²")
    ax.axhline(base_final + injected * gates.IMU_BIAS_MIN_FRACTION, ls=":", color="C3", lw=1.2,
               label=f"gate: baseline {base_final:+.4f} + "
                     f"{gates.IMU_BIAS_MIN_FRACTION:.0%} of injected")
    ax.set_ylabel("accel bias x [m/s²]")
    ax.set_title(f"IMU bias — final estimate {final:+.4f} m/s² ({fraction:.0%} of the injected "
                 f"{injected:+.3f} in absolute terms)\ngate is the differential vs the baseline: "
                 f"{final:+.4f} − {base_final:+.4f} = {diff_fraction:.1%} "
                 f"(≥ {gates.IMU_BIAS_MIN_FRACTION:.0%}, same sign)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    tb2, eb = ego_err(base)
    td2, ed = ego_err(run)
    ax2.plot(tb2, eb, "-", color="0.55", lw=1.2,
             label=f"baseline (RMSE {gates.ego_rmse(base):.3f} m)")
    ax2.plot(td2, ed, "-", color="C3", lw=1.3,
             label=f"imu_bias (RMSE {gates.ego_rmse(run):.3f} m)")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("ego position error [m]")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    return _save(fig, "imu_bias", [_gate_text("imu_bias", run, base)], images)


def fig_maneuver(runs: dict, images: Path) -> Path:
    run, base = runs["maneuver"], runs["baseline"]
    target = int(_param(run, "maneuver_target", -1))
    t_frames = frames_s(run)
    # targets.py and the gate both take the onset relative to the FIRST detection frame.
    onset = float(t_frames[0]) + float(_param(run, "maneuver_start_s", float("nan")))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                  gridspec_kw={"height_ratios": [1, 2]})

    tid = _track_for_target(run, target) if target >= 0 else None
    if tid is None:
        ax.set_title(f"target {target} was never matched to a track — nothing to plot")
    else:
        ts, probs = _mode_series(run, tid)
        for i, label in enumerate(_mode_labels(probs.shape[1] if probs.ndim == 2 else 0)):
            ax.plot(ts, probs[:, i], "-", lw=1.3, label=label)
        if probs.ndim == 2 and probs.shape[1] > gates.FIRST_CT_MODE:
            ax.plot(ts, probs[:, gates.FIRST_CT_MODE:].sum(axis=1), "-", color="k", lw=2.0,
                    label="CT total (what the gate checks)")
        if np.isfinite(onset):
            ax.axvline(onset, ls="--", color="C3", lw=1.4, label=f"onset {onset:.2f} s")
            k0 = int(np.argmax(t_frames >= onset)) if bool((t_frames >= onset).any()) else 0
            k1 = min(k0 + gates.MANEUVER_MAX_FRAMES, t_frames.size - 1)
            ax.axvspan(t_frames[k0], t_frames[k1], **SHADE)
        ax.axhline(gates.MANEUVER_CT_MIN, ls=":", color="0.3", lw=1.0)
        ax.axhline(gates.MANEUVER_CV_MAX, ls=":", color="0.3", lw=1.0)
        # Explicit newline, not tight_layout: tight_layout reserves title HEIGHT, never width,
        # so a one-line version of this ran off both ends of the 1000 px canvas and the
        # rendered PNG stopped mid-number.
        ax.set_title(f"Maneuver — track {tid} on target {target}: CT must exceed "
                     f"{gates.MANEUVER_CT_MIN:g} within {gates.MANEUVER_MAX_FRAMES} frames "
                     f"of the onset (shaded)\n"
                     f"while CV drops below {gates.MANEUVER_CV_MAX:g} at the CT peak")
        ax.legend(loc="best", fontsize=8, ncol=3)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("IMM mode probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    truth = np.asarray(run["target_truth_enu"], dtype=float)
    for i in range(truth.shape[1]):
        style = {"color": "C2", "lw": 2.2} if i == target else {"color": "0.75", "lw": 1.2}
        ax2.plot(truth[:, i, 0], truth[:, i, 1], "--", **style)
    paths = _track_paths(run)
    for other, path in sorted(paths.items()):
        if other != tid:
            ax2.plot(path[:, 1], path[:, 2], "-", color="0.75", lw=0.9)
    if tid is not None and tid in paths:
        path = paths[tid]
        ax2.plot(path[:, 1], path[:, 2], "-", color="C0", lw=1.4, marker="o", ms=3)
        after = path[path[:, 0] >= onset] if np.isfinite(onset) else path[:0]
        if after.size:
            ax2.plot(after[0, 1], after[0, 2], "*", color="C3", ms=15, zorder=5)
    ax2.plot([], [], "--", color="C2", lw=2.2, label=f"target {target} truth (turns)")
    ax2.plot([], [], "--", color="0.75", lw=1.2, label="other targets / tracks")
    ax2.plot([], [], "-", color="C0", marker="o", ms=3, label=f"track {tid}")
    ax2.plot([], [], "*", color="C3", ms=12, ls="none", label="first frame after the onset")
    ax2.set_xlabel("East [m]")
    ax2.set_ylabel("North [m]")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="datalim")

    # run_stage6.py hands every non-baseline launch the baseline npz, so pass it here too: the
    # footer must be the verdict the live gate actually reached, not a re-derived one.
    return _save(fig, "maneuver", [_gate_text("maneuver", run, base)], images)


def fig_det_dropout(runs: dict, images: Path) -> Path:
    short, coast = runs["det_dropout_short"], runs["det_dropout_coast"]
    baseline = runs.get("baseline")
    window = _window(short, "det_window") or _window(coast, "det_window")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # max_age is the ONE parameter this pair of runs is about, so it is read from the run
    # rather than written into the legend by hand — a hardcoded "max_age=15" would keep
    # claiming 15 after the preset was changed. It is a tracker_node parameter and
    # pipeline_replay records only its own, so the marker below is expected to show until the
    # recorder carries it; "(assumed)" says the figure is quoting the preset, not the run.
    styles = [(short, f"det_dropout_short ({_max_age(short, 2)})", "C3", "o"),
              (coast, f"det_dropout_coast ({_max_age(coast, 15)})", "C0", "x")]
    for run, label, color, marker in styles:
        for _target, rows in sorted(_matched_ids(run).items()):
            ax.plot(rows[:, 0], rows[:, 1], marker, color=color, ms=5, ls="none", alpha=0.8)
        ax.plot([], [], marker, color=color, ms=6, ls="none",
                label=f"{label} — {gates.id_switches(run)} ID switches")
    _shade(ax, window, label="detections suppressed")
    ax.set_ylabel("matched track id")
    ax.set_xlabel("t [s]")
    ax.set_title("Detection dropout — the same 1 s gap under two max_age budgets")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    for run, label, color, _marker in styles:
        t, err, _cnt = track_err(run)
        ax2.plot(t, err, "-", color=color, lw=1.3, label=label)
        if window is None:
            continue
        # The gate resolves re-acquisition per target; annotate exactly the frames it scored.
        # `gates.across_gap` is called by name, with no getattr fallback: a rename must be a
        # loud AttributeError here, not a figure that silently loses its whole annotation
        # layer while the title still promises it and the footer still says PASS.
        for target, (id_before, id_after, gap_err, k) in sorted(
                gates.across_gap(run, *window).items()):
            ax2.plot(t[k], gap_err, "o", color=color, ms=9, mfc="none", mew=2, zorder=4)
            ax2.annotate(f"t{target}: {id_before}→{id_after}, {gap_err:.2f} m",
                         (t[k], gap_err), textcoords="offset points", xytext=(8, 6),
                         color=color, fontsize=8)
    if baseline is not None:
        tb, errb, _ = track_err(baseline)
        ax2.plot(tb, errb, "-", color="0.60", lw=1.0, label="baseline")
    _shade(ax2, window)
    ax2.axhline(gates.COAST_REACQUIRE_MAX_M, ls="--", color="k", lw=1,
                label=f"coast re-acquisition gate {gates.COAST_REACQUIRE_MAX_M:g} m")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("mean matched track error [m]")
    ax2.set_title("Re-acquisition after the gap: id before → after, and the error at the "
                  "first post-gap frame")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    return _save(fig, "det_dropout",
                 [_gate_text("det_dropout_short", short, baseline),
                  _gate_text("det_dropout_coast", coast, baseline)], images)


def fig_clutter(runs: dict, images: Path) -> Path:
    run, base = runs["clutter"], runs["baseline"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    t, tb = frames_s(run), frames_s(base)
    dc = np.asarray(run["det_count"], dtype=float).reshape(-1)
    db = np.asarray(base["det_count"], dtype=float).reshape(-1)
    ax.step(tb, db, where="mid", color="0.55", lw=1.2, label="baseline detections")
    ax.step(t, dc, where="mid", color="C3", lw=1.4, label="clutter detections")
    n = min(t.size, tb.size)
    ax.fill_between(t[:n], db[:n], dc[:n], where=dc[:n] >= db[:n], step="mid",
                    color="C3", alpha=0.20, label="clutter excess")
    ax.set_ylabel("detections per frame")
    ax.set_title(f"Clutter — {dc.mean():.1f} detections/frame vs baseline {db.mean():.1f} "
                 f"(λ = {_param(run, 'clutter_lambda', float('nan'))})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2.step(tb, np.asarray(base["track_count"], dtype=float).reshape(-1), where="mid",
             color="0.55", lw=1.2, label="baseline tracks")
    ax2.step(t, np.asarray(run["track_count"], dtype=float).reshape(-1), where="mid",
             color="C0", lw=1.4, label="clutter tracks")
    n_ids, n_ids_base = len(gates.confirmed_track_ids(run)), len(gates.confirmed_track_ids(base))
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("tracks per frame")
    ax2.set_title(f"{n_ids} confirmed track ids vs baseline {n_ids_base}; "
                  f"{gates.id_switches(run)} ID switches vs baseline "
                  f"{gates.id_switches(base)} (both gated at baseline + {gates.CLUTTER_SLACK})")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    return _save(fig, "clutter", [_gate_text("clutter", run, base)], images)


# figure name -> (builder, the modes whose npz it needs)
FIGURES = {
    "baseline": (fig_baseline, ("baseline",)),
    "gps_dropout": (fig_gps_dropout, ("gps_dropout", "baseline")),
    "imu_bias": (fig_imu_bias, ("imu_bias", "baseline")),
    "maneuver": (fig_maneuver, ("maneuver", "baseline")),
    "det_dropout": (fig_det_dropout, ("det_dropout_short", "det_dropout_coast", "baseline")),
    "clutter": (fig_clutter, ("clutter", "baseline")),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Render the Stage 6 figures from data/cache/stage6_*.npz (host-side).")
    ap.add_argument("--only", metavar="FIGURE", choices=sorted(FIGURES),
                    help=f"render one figure: {', '.join(sorted(FIGURES))}")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                    help=f"where the recorded npz files live (default {DEFAULT_CACHE})")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_IMAGES,
                    help=f"where the PNGs are written (default {DEFAULT_IMAGES})")
    args = ap.parse_args(argv)

    wanted = [args.only] if args.only else list(FIGURES)
    needed = sorted({mode for name in wanted for mode in FIGURES[name][1]})
    runs = {mode: load_run(mode, args.cache_dir) for mode in needed}

    skipped: list[str] = []
    for name in wanted:
        builder, required = FIGURES[name]
        missing = [m for m in required if runs.get(m) is None]
        if missing:
            print(f"skipping stage6_{name}.png — no recorded run for {', '.join(missing)} "
                  f"(expected {args.cache_dir}/stage6_{missing[0]}.npz)", file=sys.stderr)
            skipped.append(name)
            continue
        builder({m: runs[m] for m in required}, args.out_dir)

    if skipped:
        print(f"\n{len(skipped)} figure(s) not rendered: {', '.join(skipped)}. "
              f"Run scripts/run_stage6.py in the container first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
