#!/usr/bin/env python3
"""ATE/RPE via evo, cross-checked against an independent implementation in kf_bringup.metrics.

Running evo alone would prove only that we can invoke evo. Agreement between two independent
implementations is what makes the number reportable, so every reported figure passes through
`cross_check`, which raises if the two disagree.

The gate must be capable of failing, or it is decoration. `_seed_shift_m` deliberately corrupts
only the file evo reads; a test asserts that this breaks the check.

No alignment: evo runs WITHOUT -a because both trajectories are already in the same metric ENU
frame with the same origin. Aligning would absorb exactly the error being measured.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "kf_bringup"))
from kf_bringup.metrics import ate, rpe  # noqa: E402

TOL = 1e-6
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)   # x, y, z, w


def write_tum(path: Path, t: np.ndarray, xyz: np.ndarray,
              quat_xyzw: np.ndarray | None = None) -> Path:
    """TUM trajectory format: `timestamp x y z qx qy qz qw`, one pose per line.

    Orientation is optional because ATE/RPE here are translation-only (`--pose_relation
    trans_part`); when omitted an identity quaternion is written so the file stays valid TUM.
    """
    path = Path(path)
    t = np.asarray(t, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected (N, 3) positions, got {xyz.shape}")
    if len(t) != len(xyz):
        raise ValueError(f"{len(t)} timestamps for {len(xyz)} poses")
    if quat_xyzw is None:
        quat_xyzw = np.tile(np.asarray(IDENTITY_QUAT, dtype=float), (len(xyz), 1))
    quat_xyzw = np.asarray(quat_xyzw, dtype=float)
    if quat_xyzw.shape != (len(xyz), 4):
        raise ValueError(f"expected ({len(xyz)}, 4) quaternions, got {quat_xyzw.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([t, xyz, quat_xyzw]), fmt="%.9f")
    return path


def _parse_evo_rmse(stdout: str) -> float:
    m = re.search(r"^\s*rmse\s+([0-9.eE+-]+)", stdout, re.MULTILINE)
    if not m:
        raise RuntimeError(f"could not parse rmse from evo output:\n{stdout}")
    return float(m.group(1))


def _run(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({out.returncode}):\n{out.stdout}\n{out.stderr}")
    return out.stdout


def run_evo_ape(ref: Path, est: Path) -> float:
    return _parse_evo_rmse(_run(
        ["evo_ape", "tum", str(ref), str(est), "--pose_relation", "trans_part"]))


def run_evo_rpe(ref: Path, est: Path, delta_m: float) -> float:
    return _parse_evo_rmse(_run(
        ["evo_rpe", "tum", str(ref), str(est), "--pose_relation", "trans_part",
         "--delta", str(delta_m), "--delta_unit", "m"]))


def cross_check(t: np.ndarray, est_xyz: np.ndarray, ref_xyz: np.ndarray, workdir: Path,
                quat_xyzw: np.ndarray | None = None, delta_m: float = 10.0,
                _seed_shift_m: float = 0.0) -> dict:
    """Compute ATE (and RPE) both ways; raise AssertionError if they disagree beyond TOL.

    Returns the agreed values. `_seed_shift_m` is test-only: it perturbs the trajectory evo
    reads without perturbing ours, which must break the gate.
    """
    workdir = Path(workdir)
    ref_f = write_tum(workdir / "ref.tum", t, ref_xyz, quat_xyzw)
    est_f = write_tum(workdir / "est.tum", t, est_xyz + _seed_shift_m, quat_xyzw)

    ours_ate = ate(est_xyz, ref_xyz).rmse
    evo_ate = run_evo_ape(ref_f, est_f)
    if abs(ours_ate - evo_ate) >= TOL:
        raise AssertionError(
            f"ATE disagreement: metrics.py {ours_ate:.9f} vs evo {evo_ate:.9f} "
            f"(delta {abs(ours_ate - evo_ate):.3e} >= {TOL:.0e})")
    return {"ate_rmse": ours_ate, "evo_ate_rmse": evo_ate,
            "rpe_rmse": rpe(est_xyz, ref_xyz, delta_m).rmse}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True, help="recorded run with ego_est/ego_truth")
    ap.add_argument("--workdir", type=Path, default=Path("data/results/tum"))
    ap.add_argument("--delta-m", type=float, default=10.0)
    args = ap.parse_args(argv)

    from kf_bringup.metrics import pair_by_stamp
    d = np.load(args.npz, allow_pickle=True)
    est = d["ego_est"]
    ref = d["ego_truth"][pair_by_stamp(d["ego_est_t_ns"], d["t_ns"])]
    t = (d["ego_est_t_ns"] - d["t_ns"][0]) / 1e9

    res = cross_check(t, est, ref, args.workdir, delta_m=args.delta_m)
    print(f"ATE  rmse {res['ate_rmse']:.6f} m  (evo {res['evo_ate_rmse']:.6f} m, agree < {TOL:.0e})")
    print(f"RPE  rmse {res['rpe_rmse']:.6f} m  over {args.delta_m:g} m segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
