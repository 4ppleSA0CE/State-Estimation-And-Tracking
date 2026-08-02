#!/usr/bin/env python3
"""Multi-sequence validation sweep: eight repaired KITTI Raw OXTS drives through the C++ pipeline
and through the frozen Python filters, into one summary table and one LaTeX table.

SELECTION BIAS -- READ THIS BEFORE QUOTING ANY NUMBER FROM HERE.
The eight drives are REPAIRED, not raw. Out-of-order OXTS samples were dropped, and where a drive
still contained an outage longer than 200 ms only its LONGEST CONTINUOUS SEGMENT is replayed.
`0009_clean` keeps 58.9% of its original wall clock and `0117_clean` keeps 52.8%; the other six
keep at least 99.6%. The kept window is by construction each drive's longest outage-free stretch,
which is exactly where an inertial filter looks its best -- no coasting through a data gap, no
timestamp inversion. Every ATE below is therefore an optimistic figure for the two heavily
trimmed drives. The retention column is printed and tabulated beside every result so this cannot
be read past silently.

What each column is measured on, because the sweep runs two different implementations:
  * ate_eskf_3d / ate_eskf_horiz / rpe_eskf / nees_pct_in_band  -- the C++ ROS 2 pipeline
    (eskf_node), replayed headless in Docker and recorded to an npz. This is the deliverable.
  * ate_ukf_3d          -- the frozen Python UKF (ukf_kitti.compare_filters), same cache, same
                           seed, same GPS noise stream, for the EKF-vs-UKF comparison.
  * nis_pct_in_band     -- the frozen Python ErrorStateEKF driven one step at a time, because the
                           PRE-update covariance never crosses the ROS wire and NIS needs it.
  * ate_gps_only        -- the raw 10 Hz noisy fixes the filter was given, against truth. This is
                           the number every ATE must be read against; without it an ATE of 0.42 m
                           is unanchored.
  * runtime_ms_per_step -- the frozen Python ESKF's measured predict+update cost per OXTS sample.
                           NOT the C++ node's: nothing in the C++ path is instrumented, and the
                           container wall clock is dominated by DDS discovery and drain.

Every ATE reported here has passed scripts/eval_ate_rpe.cross_check, which recomputes it with evo
and raises if the two implementations disagree by 1e-6 m or more. A sequence whose cross-check
fails is EXCLUDED and named in the printed summary; it is never quietly dropped.

Estimate and truth are paired by EXACT int64-nanosecond STAMP, never by index: eskf_node publishes
the state for t_k only when the IMU sample at t_{k+1} arrives, so the estimate series is one
sample short and index pairing shifts the whole trajectory.

Run from the repo root (the cache and results paths are relative to it):
    python3 scripts/run_validation.py                 # full sweep, all eight drives
    python3 scripts/run_validation.py --only 2011_09_26_0001_clean
    python3 scripts/run_validation.py --skip-pipeline # reuse the npz files already on disk
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "prototypes" / "python"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "kf_bringup"))

from eskf import (  # noqa: E402
    EskfConfig,
    ErrorStateEKF,
    build_gps_measurements,
    run_eskf,
)
from eval_ate_rpe import cross_check  # noqa: E402
from imu_mechanization import initial_state_from_oxts, mechanization_input_from_oxts  # noqa: E402
from kf_bringup.metrics import ate, chi2_band, nees, pair_by_stamp, percent_in_band, rpe  # noqa: E402
from kitti_highrate_loader import (  # noqa: E402
    HighRateOxtsConfig,
    cache_path_for,
    load_highrate_oxts,
)
from ukf_kitti import UkfConfig, compare_filters  # noqa: E402

# numpy's Accelerate BLAS backend on macOS raises spurious divide-by-zero/overflow RuntimeWarnings
# out of plain matmul. They are not real: every array this script reports is asserted finite in
# `_finite` before it reaches a CSV. Silencing only this one class keeps the sweep's output
# readable without hiding an actual non-finite result, which _finite would raise on.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

RESULTS_DIR = ROOT / "data" / "results"
CACHE_DIR = ROOT / "data" / "cache"
REPAIR_CSV = RESULTS_DIR / "oxts_repair.csv"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
LATEX_TABLE = ROOT / "docs" / "writeup" / "results_table.tex"
COMPOSE_FILE = ROOT / "docker" / "docker-compose.yml"

GPS_DOF = 3          # a GPS fix is a 3-vector, so NIS has 3 degrees of freedom
POS_DOF = 3          # position-only NEES; the recording carries no velocity or attitude truth
RPE_DELTA_M = 10.0   # RPE segment length
ALPHA = 0.05         # two-sided chi-squared band
PIPELINE_TIMEOUT_S = 1800.0


@dataclass(frozen=True)
class Sequence:
    """One repaired drive. `drive` carries the repair suffix the cache was written under."""

    date: str
    drive: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{4}_\d{2}_\d{2}", self.date):
            raise ValueError(f"date must be YYYY_MM_DD, got {self.date!r}")
        # One repair of a plain four-digit drive, never a repair of an already-derived slice:
        # `0009_tail_clean` would be a hand-picked sub-window of a sub-window.
        if not re.fullmatch(r"\d{4}_clean", self.drive):
            raise ValueError(f"drive must be a single repair of a 4-digit drive, got {self.drive!r}")

    @property
    def name(self) -> str:
        return f"{self.date}_{self.drive}"


SEQUENCES: tuple[Sequence, ...] = (
    Sequence("2011_09_26", "0001_clean"),
    Sequence("2011_09_26", "0009_clean"),
    Sequence("2011_09_26", "0015_clean"),
    Sequence("2011_09_26", "0117_clean"),
    Sequence("2011_09_30", "0020_clean"),
    Sequence("2011_09_30", "0033_clean"),
    Sequence("2011_10_03", "0042_clean"),
    Sequence("2011_09_29", "0004_clean"),
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "seq",
    "n_samples",
    "duration_s",
    "path_len_m",
    "retention_percent",
    "max_intra_segment_gap_s",
    "ate_eskf_3d",
    "ate_eskf_horiz",
    "ate_ukf_3d",
    "ate_gps_only",
    "rpe_eskf",
    "nees_pct_in_band",
    "nis_pct_in_band",
    "runtime_ms_per_step",
)

# The summary CSV carries all fourteen columns; the printed table is two-column 10pt, so the
# LaTeX version drops path length and the intra-segment gap (both still in the CSV).
LATEX_COLUMNS: tuple[str, ...] = (
    "seq", "n_samples", "duration_s", "retention_percent", "ate_eskf_3d", "ate_eskf_horiz",
    "ate_ukf_3d", "ate_gps_only", "rpe_eskf", "nees_pct_in_band", "nis_pct_in_band",
    "runtime_ms_per_step",
)

LATEX_HEADERS: dict[str, str] = {
    "seq": "Sequence",
    "n_samples": "$N$",
    "duration_s": "Dur.\\,(s)",
    "retention_percent": "Ret.\\,(\\%)",
    "ate_eskf_3d": "ATE 3D",
    "ate_eskf_horiz": "ATE hor.",
    "ate_ukf_3d": "UKF 3D",
    "ate_gps_only": "GPS only",
    "rpe_eskf": "RPE",
    "nees_pct_in_band": "NEES\\,(\\%)",
    "nis_pct_in_band": "NIS\\,(\\%)",
    "runtime_ms_per_step": "ms/step",
}

_FORMATS: dict[str, str] = {
    "n_samples": "{:d}",
    "duration_s": "{:.2f}",
    "path_len_m": "{:.1f}",
    "retention_percent": "{:.2f}",
    "max_intra_segment_gap_s": "{:.3f}",
    "ate_eskf_3d": "{:.4f}",
    "ate_eskf_horiz": "{:.4f}",
    "ate_ukf_3d": "{:.4f}",
    "ate_gps_only": "{:.4f}",
    "rpe_eskf": "{:.4f}",
    "nees_pct_in_band": "{:.1f}",
    "nis_pct_in_band": "{:.1f}",
    "runtime_ms_per_step": "{:.4f}",
}

_GATE_RE = re.compile(r"GATE\s+(\w+)\s*:\s*(PASS|FAIL)\b")


# ---------------------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------------------
def cache_path(seq: Sequence) -> Path:
    """The OXTS cache for this drive, named by the FROZEN loader, never string-built here."""
    return cache_path_for(HighRateOxtsConfig(date=seq.date, drive=seq.drive,
                                             cache_root=CACHE_DIR))


def _rel(path: Path) -> str:
    """Repo-relative for readability, absolute when the path is outside the repo (a --workdir
    elsewhere, or a tmp_path under test). Path.relative_to raises rather than falling back."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _finite(name: str, array: np.ndarray) -> np.ndarray:
    a = np.asarray(array, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains {int((~np.isfinite(a)).sum())} non-finite values")
    return a


def path_length_m(xyz: np.ndarray) -> float:
    """Total travelled arc length of a position series."""
    p = np.asarray(xyz, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (N, 3) positions, got {p.shape}")
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def horizontal_ate(est: np.ndarray, ref: np.ndarray) -> float:
    """ATE RMSE over the east/north plane only.

    Reported beside the 3D figure because the failure-mode figures elsewhere in this repo are
    horizontal-only and the two differ materially -- on a recorded baseline run the 3D 0.5673 m
    decomposes into 0.4189 m horizontal and 0.3825 m vertical.
    """
    e = np.asarray(est, dtype=float)
    r = np.asarray(ref, dtype=float)
    if e.shape != r.shape or e.ndim != 2 or e.shape[1] != 3:
        raise ValueError(f"expected matching (N, 3) arrays, got {e.shape} and {r.shape}")
    return float(np.sqrt(np.mean(np.sum((e[:, 0:2] - r[:, 0:2]) ** 2, axis=1))))


def gps_only_ate(truth_enu: np.ndarray, gps_std_m: float = 0.75, divisor: int = 10,
                 seed: int = 0) -> float:
    """ATE of the raw noisy GPS fixes against truth -- the error the filter has to beat.

    Delegates to the FROZEN eskf.build_gps_measurements, which is the same call pipeline_replay
    makes (indices first, then ONE rng.normal of shape (len(indices), 3)). Reimplementing the
    synthesis here would silently drift from the fixes the pipeline actually published the day
    somebody reorders those two lines.
    """
    truth = np.asarray(truth_enu, dtype=float)
    idx, z = build_gps_measurements(truth, divisor, gps_std_m, seed)
    return float(ate(z, truth[idx]).rmse)


def gps_update_indices(n_samples: int, config: EskfConfig) -> np.ndarray:
    """OXTS indices at which the filter actually applies a GPS update.

    Index 0 is excluded: build_gps_measurements emits a fix there, but the replay loop starts at
    k = 1, so no update is ever applied to it.
    """
    idx = np.arange(0, int(n_samples), config.gps_rate_divisor)
    return idx[idx >= 1]


def nis_series(sequence: object, config: EskfConfig, seed: int) -> np.ndarray:
    """Per-update normalised innovation squared, read BEFORE the update is applied.

    The pre-update covariance never crosses the ROS wire -- /ego/state carries only the posterior
    -- so NIS cannot be recovered from any recording. It has to be taken by driving the frozen
    ErrorStateEKF one step at a time, which this loop does in run_eskf's exact order so the
    trajectory it walks is bit-identical to the one run_eskf reports.

    Depends on two private methods of the frozen filter. That dependency is pinned by
    scripts/tests/test_run_validation.py, which asserts the Jacobian is 3x15 and equals the
    analytic [I3 | 0] for the default zero lever arm.
    """
    samples = mechanization_input_from_oxts(sequence)
    t = np.asarray(samples.timestamps, dtype=float)
    n = int(t.shape[0])

    idx, z = build_gps_measurements(
        np.asarray(sequence.enu_position_m, dtype=float),
        config.gps_rate_divisor, config.gps_std_m, seed)
    lookup = {int(i): z[j] for j, i in enumerate(idx)}

    filt = ErrorStateEKF(initial_state_from_oxts(sequence, 0), config)
    gps_cov = config.gps_covariance()

    out: list[float] = []
    for k in range(1, n):
        filt.predict(samples.accel_body[k - 1], samples.gyro_body[k - 1], float(t[k] - t[k - 1]))
        if k not in lookup:
            continue
        h = filt._measurement_jacobian()                    # 3x15
        nu = lookup[k] - filt._measurement_prediction()     # innovation
        s = h @ filt.P @ h.T + gps_cov                      # filt.P is still PRE-update here
        out.append(float(nu @ np.linalg.solve(s, nu)))
        filt.update_gps(lookup[k])
    return np.asarray(out, dtype=float)


def repair_lookup(csv_path: Path | str = REPAIR_CSV) -> dict[tuple[str, str], dict[str, str]]:
    """Retention and worst intra-segment gap per repaired drive, keyed by (date, clean_drive).

    Those two columns are produced by a DIFFERENT script. If they are not in the file, the cells
    come back empty and stay empty all the way into the table -- a fabricated 100% retention
    would be worse than a visibly missing one.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (str(row.get("date", "")).strip(), str(row.get("clean_drive", "")).strip())
            out[key] = {
                "retention_percent": str(row.get("retention_percent", "") or "").strip(),
                "max_intra_segment_gap_s": str(row.get("max_intra_segment_gap_s", "") or "").strip(),
            }
    return out


def summary_row(values: dict) -> dict:
    """Order and validate one summary row: EXACTLY SUMMARY_COLUMNS, no more, no fewer."""
    missing = [c for c in SUMMARY_COLUMNS if c not in values]
    extra = [c for c in values if c not in SUMMARY_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"summary row column mismatch: missing {missing}, unknown {extra}")
    return {c: values[c] for c in SUMMARY_COLUMNS}


def _cell(column: str, value) -> str:
    """Render one cell. An empty value stays empty in the CSV -- never a substituted zero."""
    if value is None or value == "":
        return ""
    fmt = _FORMATS.get(column)
    if fmt is None:
        return str(value)
    try:
        return fmt.format(int(value) if fmt.endswith("d}") else float(value))
    except (TypeError, ValueError):
        return str(value)


def _tex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def latex_table(rows: list[dict], columns: tuple[str, ...] = LATEX_COLUMNS) -> str:
    """A booktabs table over `rows`, generated -- never hand-typed -- from the summary rows."""
    header = " & ".join(LATEX_HEADERS.get(c, _tex_escape(c)) for c in columns)
    body = []
    for row in rows:
        cells = []
        for c in columns:
            text = _cell(c, row.get(c, ""))
            # An empty cell in a LaTeX row renders as blank space, which reads as an oversight.
            # An explicit dash says "this was not measured".
            cells.append(_tex_escape(text) if text else "--")
        body.append(" & ".join(cells) + " \\\\")
    align = "l" + "r" * (len(columns) - 1)
    return "\n".join([
        "% GENERATED by scripts/run\\_validation.py from data/results/summary.csv. Do not edit.",
        "\\begin{table*}[t]",
        "\\centering\\footnotesize",
        "\\caption{Validation sweep over eight repaired KITTI Raw drives. ATE and RPE are "
        "unaligned and in metres; RPE uses 10\\,m segments. ATE 3D, ATE hor., RPE and NEES come "
        "from the C++ pipeline; UKF 3D and ms/step come from the Python reference filters on the "
        "same cache and the same GPS noise stream. NEES is position-only (3 DOF) against the "
        "filter's own covariance; NIS is the 3 DOF GPS innovation, both as the percentage of "
        "steps inside the 95\\% $\\chi^2$ band. \\emph{Ret.} is the fraction of each drive's "
        "original wall clock that survived repair: the replayed window is that drive's longest "
        "outage-free segment, which flatters the two drives below 60\\%.}",
        "\\label{tab:validation}",
        # Twelve columns at the default 6pt tabcolsep overfull the full-width table* by 50.9pt
        # (measured). 3.5pt reclaims 60pt across the 24 inter-column gaps. Scoped inside the
        # table environment, so no other table in the document is affected.
        "\\setlength{\\tabcolsep}{3.5pt}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]) + "\n"


# ---------------------------------------------------------------------------------------
# The C++ pipeline run.
# ---------------------------------------------------------------------------------------
def _container_path(host_path: Path) -> str:
    return "/workspace/" + Path(host_path).resolve().relative_to(ROOT).as_posix()


def pipeline_npz(seq: Sequence) -> Path:
    return CACHE_DIR / f"validation_{seq.name}.npz"


def run_pipeline(seq: Sequence, timeout_s: float = PIPELINE_TIMEOUT_S) -> dict:
    """Replay one drive through the C++ pipeline headless and record it.

    `foxglove` defaults false, so the lidar/image/viz nodes never start and no Velodyne or camera
    data is needed -- these repaired drives ship oxts only.

    The launch EXIT CODE is deliberately not the success criterion. pipeline_replay exits 1 when
    the baseline gate fails, and that gate is tuned for drive_0001's synthetic targets; on a
    different drive it can fail while the ego recording is perfectly good. Success here is "the
    npz exists and carries an ego series". The gate verdict is captured and reported alongside.
    """
    out_npz = pipeline_npz(seq)
    inner = (
        "cd /workspace/ros2_ws && source install/setup.bash && "
        "ros2 launch kf_bringup full_pipeline.launch.py mode:=baseline "
        f"cache_path:={_container_path(cache_path(seq))} "
        f"output_npz:={_container_path(out_npz)}"
    )
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "run", "--rm", "dev", "bash", "-lc", inner]

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        code, out, timed_out = proc.returncode, proc.stdout + proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, timed_out = -1, True
        out = (exc.stdout or "") + (exc.stderr or "") if isinstance(exc.stdout, str) else ""
    seconds = time.monotonic() - t0

    hits = _GATE_RE.findall(out)
    verdict = next((r for name, r in reversed(hits) if name == "baseline"), "MISSING")
    return {
        "npz": out_npz, "code": code, "seconds": seconds, "timed_out": timed_out,
        "gate_verdict": verdict,
        "gate_lines": [ln[ln.index("GATE"):].rstrip() for ln in out.splitlines() if "GATE" in ln],
        "tail": "\n".join(out.splitlines()[-30:]),
    }


# ---------------------------------------------------------------------------------------
# Per-sequence evaluation.
# ---------------------------------------------------------------------------------------
def evaluate(seq: Sequence, repair: dict[tuple[str, str], dict[str, str]],
             workdir: Path) -> dict:
    """Every metric for one sequence. Raises on anything that would make a number unreportable."""
    sequence = load_highrate_oxts(HighRateOxtsConfig(date=seq.date, drive=seq.drive,
                                                     cache_root=CACHE_DIR))
    truth = _finite("truth", sequence.enu_position_m)
    n = int(sequence.sample_count)

    # --- the C++ pipeline recording -----------------------------------------------------
    npz_path = pipeline_npz(seq)
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} does not exist; the pipeline run did not record it")
    rec = np.load(npz_path, allow_pickle=True)
    if "ego_cov" not in rec.files:
        raise KeyError(f"{npz_path} has no ego_cov; it predates the covariance recording")
    est = _finite("ego_est", rec["ego_est"])
    if len(est) == 0:
        raise ValueError(f"{npz_path} recorded no /ego/state samples")
    ref_idx = pair_by_stamp(rec["ego_est_t_ns"], rec["t_ns"])   # EXACT stamp, never index
    ref = _finite("ego_truth", rec["ego_truth"])[ref_idx]
    est_t = (rec["ego_est_t_ns"] - rec["t_ns"][0]) / 1e9
    cov = _finite("ego_cov", rec["ego_cov"]).reshape(len(est), 15, 15)

    # --- the frozen Python filters on the SAME cache -------------------------------------
    cmp_result = compare_filters(sequence, UkfConfig(), seed=0)
    eskf_py = cmp_result["eskf_clean"]
    ukf_py = cmp_result["ukf_clean"]
    py_truth = _finite("python truth", eskf_py["truth_positions"])
    ukf_est = _finite("ukf x_est", ukf_py["x_est"])
    eskf_py_est = _finite("python eskf x_est", eskf_py["x_est"])
    seq_t = np.asarray(sequence.timestamps, dtype=float)

    # --- ATE, every one cross-checked against evo ----------------------------------------
    workdir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {}
    checks["eskf_cpp"] = cross_check(est_t, est, ref, workdir / "eskf_cpp", delta_m=RPE_DELTA_M)
    checks["ukf_py"] = cross_check(seq_t, ukf_est, py_truth, workdir / "ukf_py",
                                   delta_m=RPE_DELTA_M)
    checks["eskf_py"] = cross_check(seq_t, eskf_py_est, py_truth, workdir / "eskf_py",
                                    delta_m=RPE_DELTA_M)
    gps_idx, gps_z = build_gps_measurements(truth, EskfConfig().gps_rate_divisor,
                                            EskfConfig().gps_std_m, 0)
    checks["gps_only"] = cross_check(seq_t[gps_idx], gps_z, truth[gps_idx],
                                     workdir / "gps_only", delta_m=RPE_DELTA_M)

    # --- consistency ---------------------------------------------------------------------
    nees_pos = nees(est - ref, cov[:, 0:3, 0:3])
    nis = nis_series(sequence, EskfConfig(), seed=0)
    nis_idx = gps_update_indices(n, EskfConfig())
    if len(nis) != len(nis_idx):
        raise ValueError(f"NIS length {len(nis)} does not match {len(nis_idx)} GPS updates")

    runtime_ms = cmp_result["eskf_runtime_s"] / n * 1e3
    rec_repair = repair.get((seq.date, seq.drive),
                            {"retention_percent": "", "max_intra_segment_gap_s": ""})

    return {
        "sequence": sequence,
        "row": summary_row({
            "seq": seq.name,
            "n_samples": n,
            "duration_s": float(sequence.duration_s),
            "path_len_m": path_length_m(truth),
            "retention_percent": rec_repair["retention_percent"],
            "max_intra_segment_gap_s": rec_repair["max_intra_segment_gap_s"],
            "ate_eskf_3d": checks["eskf_cpp"]["ate_rmse"],
            "ate_eskf_horiz": horizontal_ate(est, ref),
            "ate_ukf_3d": checks["ukf_py"]["ate_rmse"],
            "ate_gps_only": checks["gps_only"]["ate_rmse"],
            "rpe_eskf": float(rpe(est, ref, RPE_DELTA_M).rmse),
            "nees_pct_in_band": percent_in_band(nees_pos, POS_DOF, ALPHA),
            "nis_pct_in_band": percent_in_band(nis, GPS_DOF, ALPHA),
            "runtime_ms_per_step": runtime_ms,
        }),
        "checks": checks,
        "ate_eskf_py_3d": checks["eskf_py"]["ate_rmse"],
        "ukf_runtime_ms_per_step": cmp_result["ukf_runtime_s"] / n * 1e3,
        "series": {
            "est_t": est_t, "est": est, "ref": ref, "nees": nees_pos,
            "seq_t": seq_t, "truth": truth, "ukf_est": ukf_est, "eskf_py_est": eskf_py_est,
            "gps_idx": gps_idx, "gps_z": gps_z, "nis": nis, "nis_t": seq_t[nis_idx],
        },
    }


def write_series_csvs(seq: Sequence, result: dict) -> list[Path]:
    """One CSV per sequence per filter: the C++ ESKF run and the Python UKF run."""
    s = result["series"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # NIS lives on the OXTS index base and the C++ estimate lives on the OXTS stamp base, so the
    # two align exactly -- but only on the GPS-update steps. Non-update rows stay blank.
    nis_at_t = dict(zip(np.round(s["nis_t"], 9), s["nis"]))

    path = RESULTS_DIR / f"{seq.name}_eskf.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "est_e", "est_n", "est_u", "truth_e", "truth_n", "truth_u",
                    "err_3d_m", "err_horiz_m", "nees_pos", "nis"])
        for k in range(len(s["est"])):
            e, r = s["est"][k], s["ref"][k]
            t = float(s["est_t"][k])
            nis_v = nis_at_t.get(round(t, 9), "")
            w.writerow([f"{t:.9f}", *(f"{v:.6f}" for v in e), *(f"{v:.6f}" for v in r),
                        f"{np.linalg.norm(e - r):.6f}", f"{np.linalg.norm(e[0:2] - r[0:2]):.6f}",
                        f"{s['nees'][k]:.6f}", f"{nis_v:.6f}" if nis_v != "" else ""])
    written.append(path)

    path = RESULTS_DIR / f"{seq.name}_ukf.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "est_e", "est_n", "est_u", "truth_e", "truth_n", "truth_u",
                    "err_3d_m", "err_horiz_m"])
        for k in range(len(s["ukf_est"])):
            e, r = s["ukf_est"][k], s["truth"][k]
            w.writerow([f"{float(s['seq_t'][k]):.9f}", *(f"{v:.6f}" for v in e),
                        *(f"{v:.6f}" for v in r), f"{np.linalg.norm(e - r):.6f}",
                        f"{np.linalg.norm(e[0:2] - r[0:2]):.6f}"])
    written.append(path)
    return written


def plot_sequence(seq: Sequence, result: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = result["series"]
    row = result["row"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(s["truth"][:, 0], s["truth"][:, 1], "k-", lw=1.6, label="OXTS truth")
    ax.scatter(s["gps_z"][:, 0], s["gps_z"][:, 1], s=6, c="gray", alpha=0.35,
               label=f"GPS fixes (ATE {row['ate_gps_only']:.3f} m)")
    ax.plot(s["est"][:, 0], s["est"][:, 1], "b-", lw=1.2,
            label=f"ESKF C++ (ATE {row['ate_eskf_3d']:.3f} m)")
    ax.plot(s["ukf_est"][:, 0], s["ukf_est"][:, 1], "r--", lw=1.0,
            label=f"UKF Python (ATE {row['ate_ukf_3d']:.3f} m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_title("Trajectory")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    err3 = np.linalg.norm(s["est"] - s["ref"], axis=1)
    errh = np.linalg.norm(s["est"][:, 0:2] - s["ref"][:, 0:2], axis=1)
    ax.plot(s["est_t"], err3, "b-", lw=1.0, label="3D")
    ax.plot(s["est_t"], errh, "g-", lw=1.0, label="horizontal")
    ax.set_xlabel("time [s]"); ax.set_ylabel("position error [m]")
    ax.set_title("C++ ESKF position error"); ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    lo, hi = chi2_band(POS_DOF, alpha=ALPHA)
    ax = axes[1, 0]
    ax.plot(s["est_t"], s["nees"], "b-", lw=0.8)
    ax.axhline(POS_DOF, color="k", ls="--", lw=1.0, label=f"E[NEES]={POS_DOF}")
    ax.axhline(lo, color="gray", ls=":", lw=1.0)
    ax.axhline(hi, color="gray", ls=":", lw=1.0, label="95% band")
    ax.set_yscale("log"); ax.set_xlabel("time [s]"); ax.set_ylabel("NEES (position, 3 DOF)")
    ax.set_title(f"NEES, {row['nees_pct_in_band']:.1f}% in band")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    lo, hi = chi2_band(GPS_DOF, alpha=ALPHA)
    ax = axes[1, 1]
    ax.plot(s["nis_t"], s["nis"], "b-", lw=0.8)
    ax.axhline(GPS_DOF, color="k", ls="--", lw=1.0, label=f"E[NIS]={GPS_DOF}")
    ax.axhline(lo, color="gray", ls=":", lw=1.0)
    ax.axhline(hi, color="gray", ls=":", lw=1.0, label="95% band")
    ax.set_yscale("log"); ax.set_xlabel("time [s]"); ax.set_ylabel("NIS (GPS, 3 DOF)")
    ax.set_title(f"NIS, {row['nis_pct_in_band']:.1f}% in band")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    ret = row["retention_percent"]
    # Two decimals, not one: 99.95% must not be rounded up to a flat "100.0%" in the one caption
    # a reader uses to judge how much of the drive was thrown away.
    ret_text = f"{float(ret):.2f}% of original wall clock retained" if ret != "" else \
        "retention not recorded"
    fig.suptitle(f"{seq.name}  --  {row['n_samples']} samples, {float(row['duration_s']):.1f} s, "
                 f"{ret_text}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{seq.name}_validation.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------------------
def write_summary_csv(rows: list[dict], path: Path = SUMMARY_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_COLUMNS)
        for row in rows:
            w.writerow([_cell(c, row[c]) for c in SUMMARY_COLUMNS])
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-sequence validation sweep.")
    ap.add_argument("--only", action="append", metavar="SEQ",
                    help="run one sequence by name, e.g. 2011_09_26_0001_clean (repeatable)")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="reuse the recorded npz files instead of re-running Docker")
    ap.add_argument("--timeout", type=float, default=PIPELINE_TIMEOUT_S,
                    help=f"per-sequence Docker wall-clock limit (default {PIPELINE_TIMEOUT_S:g} s)")
    ap.add_argument("--workdir", type=Path, default=RESULTS_DIR / "tum",
                    help="scratch directory for the evo cross-check trajectory files")
    ap.add_argument("--table-only", action="store_true",
                    help="regenerate the LaTeX table from the existing summary.csv and stop; "
                         "runs no filter and measures nothing")
    args = ap.parse_args(argv)

    if args.table_only:
        if not SUMMARY_CSV.exists():
            raise SystemExit(f"{SUMMARY_CSV} does not exist; run the sweep first")
        with SUMMARY_CSV.open(newline="") as fh:
            rows = [summary_row(dict(r)) for r in csv.DictReader(fh)]
        LATEX_TABLE.write_text(latex_table(rows))
        print(f"wrote {_rel(LATEX_TABLE)} from {_rel(SUMMARY_CSV)} "
              f"({len(rows)} row(s))")
        return 0

    selected = SEQUENCES
    if args.only:
        by_name = {s.name: s for s in SEQUENCES}
        unknown = [n for n in args.only if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown sequence(s) {unknown}; known: {sorted(by_name)}")
        selected = tuple(by_name[n] for n in args.only)

    repair = repair_lookup()
    if not repair:
        print(f"NOTE: {REPAIR_CSV} not found; retention and gap cells will be empty.")

    rows: list[dict] = []
    excluded: list[tuple[str, str]] = []
    notes: list[str] = []
    check_report: list[tuple[str, str, float, float, float]] = []
    timings: list[tuple[str, int, float]] = []

    for i, seq in enumerate(selected, start=1):
        print(f"\n=== [{i}/{len(selected)}] {seq.name} ===", flush=True)
        cache = cache_path(seq)
        if not cache.exists():
            try:
                load_highrate_oxts(HighRateOxtsConfig(date=seq.date, drive=seq.drive,
                                                      cache_root=CACHE_DIR))
            except Exception as exc:                      # noqa: BLE001 -- record and continue
                excluded.append((seq.name, f"OXTS cache could not be built: {exc}"))
                print(f"EXCLUDED {seq.name}: {exc}")
                continue
        print(f"cache {cache}")

        if not args.skip_pipeline:
            res = run_pipeline(seq, timeout_s=args.timeout)
            timings.append((seq.name, 0, res["seconds"]))
            print(f"pipeline exit={res['code']} gate={res['gate_verdict']} "
                  f"wall={res['seconds']:.1f} s -> {res['npz'].name}")
            for line in res["gate_lines"]:
                print(f"  {line}")
            if res["timed_out"]:
                excluded.append((seq.name, f"pipeline timed out after {args.timeout:.0f} s"))
                print(f"EXCLUDED {seq.name}: pipeline timed out")
                continue
            if not res["npz"].exists():
                excluded.append((seq.name,
                                 f"pipeline recorded no npz (exit {res['code']}); "
                                 f"tail:\n{res['tail']}"))
                print(f"EXCLUDED {seq.name}: no npz recorded (exit {res['code']})")
                print(res["tail"])
                continue
            if res["gate_verdict"] != "PASS":
                # Informational only: the gate scores the SYNTHETIC targets, which are tuned for
                # drive_0001. A FAIL there does not invalidate the ego recording, so the sequence
                # stays in -- but the reader is told.
                notes.append(f"{seq.name}: pipeline baseline gate {res['gate_verdict']} "
                             f"(synthetic-target gate, does not affect the ego metrics)")

        try:
            result = evaluate(seq, repair, args.workdir / seq.name)
        except AssertionError as exc:      # cross_check disagreement -- never report the number
            excluded.append((seq.name, f"evo cross-check FAILED: {exc}"))
            print(f"EXCLUDED {seq.name}: evo cross-check failed: {exc}")
            continue
        except Exception as exc:                          # noqa: BLE001 -- record and continue
            excluded.append((seq.name, f"{type(exc).__name__}: {exc}"))
            print(f"EXCLUDED {seq.name}: {type(exc).__name__}: {exc}")
            continue

        row = result["row"]
        rows.append(row)
        if timings and timings[-1][0] == seq.name:
            timings[-1] = (seq.name, int(row["n_samples"]), timings[-1][2])
        for label, chk in result["checks"].items():
            check_report.append((seq.name, label, chk["ate_rmse"], chk["evo_ate_rmse"],
                                 abs(chk["ate_rmse"] - chk["evo_ate_rmse"])))

        for path in write_series_csvs(seq, result):
            print(f"wrote {_rel(path)}")
        print(f"wrote {_rel(plot_sequence(seq, result))}")
        print(f"  ATE C++ ESKF 3D {row['ate_eskf_3d']:.4f} m | horiz {row['ate_eskf_horiz']:.4f} m"
              f" | Python ESKF 3D {result['ate_eskf_py_3d']:.4f} m"
              f" | UKF 3D {row['ate_ukf_3d']:.4f} m | GPS-only {row['ate_gps_only']:.4f} m")
        print(f"  RPE {row['rpe_eskf']:.4f} m/10 m | NEES {row['nees_pct_in_band']:.1f}% "
              f"| NIS {row['nis_pct_in_band']:.1f}% in band")

    if rows:
        print(f"\nwrote {_rel(write_summary_csv(rows))}")
        LATEX_TABLE.parent.mkdir(parents=True, exist_ok=True)
        LATEX_TABLE.write_text(latex_table(rows))
        print(f"wrote {_rel(LATEX_TABLE)}")

    print("\n=== summary ===")
    width = max([len(r["seq"]) for r in rows], default=10)
    print(f"{'seq':<{width}}  {'N':>6} {'dur_s':>8} {'ret%':>7} {'ATE3D':>8} {'ATEh':>8} "
          f"{'UKF3D':>8} {'GPS':>8} {'RPE':>8} {'NEES%':>7} {'NIS%':>7} {'ms/step':>8}")
    for r in rows:
        ret = f"{float(r['retention_percent']):.2f}" if r["retention_percent"] != "" else "--"
        print(f"{r['seq']:<{width}}  {int(r['n_samples']):>6} {float(r['duration_s']):>8.2f} "
              f"{ret:>7} {float(r['ate_eskf_3d']):>8.4f} {float(r['ate_eskf_horiz']):>8.4f} "
              f"{float(r['ate_ukf_3d']):>8.4f} {float(r['ate_gps_only']):>8.4f} "
              f"{float(r['rpe_eskf']):>8.4f} {float(r['nees_pct_in_band']):>7.1f} "
              f"{float(r['nis_pct_in_band']):>7.1f} {float(r['runtime_ms_per_step']):>8.4f}")

    print("\n=== evo cross-check (metrics.py vs evo, must agree < 1e-6 m) ===")
    for name, label, ours, theirs, delta in check_report:
        print(f"{name:<{width}}  {label:<9} ours {ours:.9f}  evo {theirs:.9f}  "
              f"delta {delta:.2e}  AGREE")

    print("\n=== SELECTION BIAS ===")
    print("These drives are REPAIRED. Out-of-order OXTS samples were removed, and where a drive "
          "had outages longer than 200 ms only its LONGEST CONTINUOUS SEGMENT is replayed.")
    print("0009_clean retains 58.9% of its original wall clock and 0117_clean retains 52.8%; the "
          "other six retain at least 99.6%.")
    print("The kept window is by construction each drive's longest outage-free stretch -- which "
          "is where an inertial filter looks its best. Read every ATE above with that in mind.")

    if notes:
        print("\n=== notes ===")
        for note in notes:
            print(f"  {note}")

    print(f"\n{len(rows)} sequence(s) in the table, {len(excluded)} excluded "
          f"of {len(selected)} attempted.")
    if excluded:
        print("=== EXCLUDED SEQUENCES ===")
        for name, why in excluded:
            print(f"  {name}: {why}")
    else:
        print("No sequence was excluded.")

    if timings:
        print("\n=== pipeline wall clock ===")
        for name, n, secs in timings:
            rate = f"{secs / n * 1e3:.3f} ms/sample" if n else "n unknown"
            print(f"  {name:<{width}}  {secs:>7.1f} s  ({rate})")

    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
