"""Write the ESKF Python reference trajectory (timestamps + x_est) for the C++ parity gate.

The C++ ESKF node (kf_eskf) is validated against this reference by kitti_replay.py::_check_parity.
The UKF reference is intentionally not generated here: the C++ UKF node is parked (see
PROJECT_PRD.md Stage 3), so there is nothing to gate against. The Python UKF comparison lives in
prototypes/python/ukf_kitti.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prototypes" / "python"))

from eskf import EskfConfig, run_eskf  # noqa: E402
from kitti_highrate_loader import (  # noqa: E402
    HighRateOxtsConfig,
    load_highrate_oxts,
    require_highrate_oxts,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = HighRateOxtsConfig()
    require_highrate_oxts(cfg)
    sequence = load_highrate_oxts(cfg)

    e = run_eskf(sequence, EskfConfig(), args.seed)
    np.savez(args.out_dir / "eskf_py_ref.npz", timestamps=e["timestamps"], x_est=e["x_est"])
    print(f"wrote {args.out_dir}/eskf_py_ref.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
