"""Drive the seven failure-mode launches of the coupled pipeline and summarise their verdicts.

Runs INSIDE the ROS2 container — it needs `ros2 launch`. The figures are drawn on the HOST by
scripts/plot_failure_modes.py, because matplotlib is not in the image. Same split that
scripts/plot_tracker_parity.py already documents.

    docker compose -f docker/docker-compose.yml run --rm dev bash -lc \\
      'cd /workspace/ros2_ws && source install/setup.bash && \\
       python3 /workspace/scripts/run_failure_modes.py'

Modes run SEQUENTIALLY and never in parallel: they share one DDS domain and the data/cache npz
paths, and six of the seven gates read pipeline_baseline.npz as their reference — so `baseline`
goes first and the rest follow. `--only <mode>` re-runs a single one against the baseline npz
already on disk.

The verdict is the `GATE <mode>: PASS|FAIL` line pipeline_replay prints, not the launch exit
code: `ros2 launch` also returns non-zero when an unrelated node crashes on teardown, and a run
that emitted no verdict line at all must never be summarised as a pass. Both are reported, and a
disagreement between them is called out rather than hidden.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# APPEND, never insert: in the container an installed kf_bringup is on the path already, and
# that install is what the launched nodes import. Putting the source tree FIRST would let this
# driver's MODES drift away from the mode list the nodes actually implement — the one
# disagreement a driver must never have. The source tree is the fallback for a host run, where
# nothing is installed.
sys.path.append(str(ROOT / "ros2_ws" / "src" / "kf_bringup"))

from kf_bringup.failure_gates import MODES  # noqa: E402  the single source of the mode list

BASELINE = "baseline"
LAUNCH_FILE = "full_pipeline.launch.py"
DEFAULT_OUT_DIR = ROOT / "data" / "cache"
TAIL_LINES = 40

# The launch prefixes every line ("[pipeline_replay-4] GATE ..."), so match anywhere.
_VERDICT_RE = re.compile(r"GATE\s+(\w+)\s*:\s*(PASS|FAIL)\b")


@dataclass
class Result:
    mode: str
    code: int
    # PASS | FAIL | MISSING (no verdict line at all) | MISMATCH (the only verdict names
    # another mode)
    verdict: str
    seconds: float
    timed_out: bool
    lines: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """A run is ok only if it reached its own PASS verdict AND was allowed to finish.

        A timeout is never ok whatever stdout said: the process group was SIGKILLed, so the
        npz the next mode's ratio gate reads may be truncated or absent even though a PASS
        line had already been printed.
        """
        return self.verdict == "PASS" and not self.timed_out


def _launch_cmd(mode: str, out_dir: Path, baseline_npz: str, extra: list[str]) -> list[str]:
    """The `ros2 launch` argv for one mode.

    The `baseline_npz` token is OMITTED entirely for the baseline run rather than passed empty:
    `ros2launch.api.parse_launch_arguments` rejects any token ending in ':=' with
    `RuntimeError: malformed launch argument`, so `baseline_npz:=` kills the launch before a
    single node starts. The launch file declares the argument with `default_value=""`, and
    `_gate_baseline` ignores its baseline argument anyway — only the token's absence matters.
    """
    cmd = [
        "ros2", "launch", "kf_bringup", LAUNCH_FILE,
        f"mode:={mode}",
        f"output_npz:={out_dir / f'pipeline_{mode}.npz'}",
    ]
    if baseline_npz:
        cmd.append(f"baseline_npz:={baseline_npz}")
    return cmd + list(extra)


def _run(cmd: list[str], timeout: float) -> tuple[int, str, bool]:
    """Run one launch to completion. Returns (exit code, combined stdout+stderr, timed out).

    start_new_session + killpg because `ros2 launch` spawns the four nodes as children: killing
    only the launcher on a timeout would leave them holding the DDS domain, and the next mode
    would then see stale publishers instead of a clean run.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return int(proc.returncode), out, False
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, _ = proc.communicate()
        return int(proc.returncode if proc.returncode is not None else -9), out, True


def _gate_lines(out: str) -> list[str]:
    """The gate's own report lines, with the launch's node prefix trimmed off."""
    return [line[line.index("GATE"):].rstrip() for line in out.splitlines() if "GATE" in line]


def _verdict(out: str, mode: str) -> str:
    """The verdict for THIS mode: PASS | FAIL | MISSING | MISMATCH.

    A verdict line naming a DIFFERENT mode is never adopted. Its usual cause is a dropped or
    defaulted `mode` parameter, i.e. a run that silently executed some other preset — exactly
    the failure this driver exists to catch, so it gets its own not-ok verdict rather than
    being reported as this mode's result.
    """
    hits = _VERDICT_RE.findall(out)
    if not hits:
        return "MISSING"
    for name, result in reversed(hits):
        if name == mode:
            return result
    return "MISMATCH"


def _verdict_names(out: str) -> list[str]:
    """Every mode name that appeared in a verdict line, for the MISMATCH diagnostic."""
    return sorted({name for name, _result in _VERDICT_RE.findall(out)})


def _tail(out: str, n: int = TAIL_LINES) -> str:
    lines = out.splitlines()
    return "\n".join(lines[-n:])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the failure-mode suite (inside the ROS2 container).")
    ap.add_argument("--only", metavar="MODE", choices=MODES,
                    help="run a single mode instead of all seven; the baseline npz must "
                         "already exist for the ratio gates")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"where the per-mode npz files are written (default {DEFAULT_OUT_DIR})")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-mode wall-clock limit in seconds (default 600)")
    ap.add_argument("--verbose", action="store_true",
                    help="echo the full launch output for every mode, not just failures")
    ap.add_argument("extra", nargs="*", metavar="key:=value",
                    help="extra launch arguments forwarded to every run, e.g. rate_scale:=1.0")
    args = ap.parse_args(argv)

    if BASELINE not in MODES:
        raise SystemExit(f"failure_gates.MODES has no {BASELINE!r} entry: {MODES!r}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_npz = out_dir / f"pipeline_{BASELINE}.npz"

    # baseline first, then the rest in MODES order — six gates read the baseline npz.
    modes = [args.only] if args.only else [BASELINE] + [m for m in MODES if m != BASELINE]
    if args.only and args.only != BASELINE and not baseline_npz.exists():
        print(f"WARNING: {baseline_npz} does not exist; the ratio gate for {args.only} will "
              f"fail for lack of a reference. Run the baseline first.", file=sys.stderr)

    results: list[Result] = []
    for i, mode in enumerate(modes, start=1):
        # "" for the baseline run: it IS the reference, and _launch_cmd then omits the token.
        ref = "" if mode == BASELINE else str(baseline_npz)
        cmd = _launch_cmd(mode, out_dir, ref, list(args.extra))
        print(f"\n=== [{i}/{len(modes)}] {mode} ===", flush=True)
        print("$ " + " ".join(cmd), flush=True)

        t0 = time.monotonic()
        code, out, timed_out = _run(cmd, args.timeout)
        res = Result(mode=mode, code=code, verdict=_verdict(out, mode),
                     seconds=time.monotonic() - t0, timed_out=timed_out,
                     lines=_gate_lines(out))
        results.append(res)

        for line in res.lines:
            print(line)
        if timed_out:
            print(f"TIMEOUT after {args.timeout:.0f} s — process group killed. This run is NOT "
                  f"a pass even if a PASS line was printed first: {out_dir}/pipeline_{mode}.npz "
                  f"may be truncated or missing.", file=sys.stderr)
        if res.verdict == "MISMATCH":
            print(f"MISMATCH: launched {mode!r} but the only GATE verdict line(s) name "
                  f"{_verdict_names(out)} — the `mode` argument was most likely dropped or "
                  f"defaulted, so this is not a result for {mode!r}.", file=sys.stderr)
        if args.verbose:
            print(out)
        elif not res.ok:
            print(f"--- last {TAIL_LINES} lines of {mode} ---\n{_tail(out)}", file=sys.stderr)
        if res.ok and code != 0:
            print(f"NOTE: {mode} reported PASS but the launch exited {code} — a node most "
                  f"likely crashed on teardown. Worth a look; the gate itself passed.",
                  file=sys.stderr)

    width = max(len(r.mode) for r in results)
    print("\n=== failure-mode summary ===")
    print(f"{'mode':<{width}}  {'exit':>4}  {'verdict':<8} {'sec':>7}")
    for r in results:
        # A timed-out run is shown as TIMEOUT, never as the verdict it happened to print
        # before it was killed — that verdict is reported alongside, not instead.
        shown = "TIMEOUT" if r.timed_out else r.verdict
        note = f"  (stdout said {r.verdict})" if r.timed_out else ""
        print(f"{r.mode:<{width}}  {r.code:>4}  {shown:<8} {r.seconds:>7.1f}{note}")

    failed = [r.mode for r in results if not r.ok]
    print(f"\n{len(results)} run(s): {len(results) - len(failed)} PASS, {len(failed)} not-PASS")

    print("\n=== gate reports ===")
    for r in results:
        print(f"[{r.mode}]")
        for line in r.lines or ["(no GATE output — the replay never reached its verdict)"]:
            print(f"  {line}")

    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
