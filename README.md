## State Estimation and Tracking

Python and MATLAB prototypes for Kalman-filter state estimation and tracking. Stage 0
uses synthetic scenarios only, so setup and smoke tests do not require KITTI data.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r prototypes/python/requirements.txt
```

The Python prototype dependencies are constrained in
`prototypes/python/requirements.txt` to keep major-version behavior stable while
allowing compatible patch updates.

## Run Python Prototypes

Each command writes generated figures and reference exports to `prototypes/output/`.
That directory is local output and is intentionally ignored by git.

```bash
python prototypes/python/linear_kf.py
python prototypes/python/ekf_synthetic.py
python prototypes/python/ukf_synthetic.py
python prototypes/python/imm_synthetic.py
```

## Test and Smoke-Check

Stage 0 tests are synthetic-only and should run without KITTI. A fresh environment can
verify importability and execute the Stage 0 smoke path with:

```bash
python -m pytest prototypes/python/tests
python -m compileall prototypes/python
python prototypes/python/linear_kf.py
python prototypes/python/ekf_synthetic.py
python prototypes/python/ukf_synthetic.py
python prototypes/python/imm_synthetic.py
```

Expected checks include NEES/NIS consistency messages for the KF/EKF/UKF scripts and
IMM RMSE/mode-switch DOD messages for the IMM script.

## MATLAB Parity Verification

Run the Python prototypes first because the MATLAB scripts load the `.mat` reference
exports generated in `prototypes/output/`. Then run, from the repository root:

```bash
matlab -batch "addpath('prototypes/matlab'); linear_kf; ekf_synthetic; ukf_synthetic; imm_synthetic"
```

Each MATLAB script reports a Python/MATLAB max-error parity check and writes any
MATLAB-generated plots to `prototypes/output/`.

## KITTI Data Policy

KITTI data is not required for Stage 0. Keep downloaded datasets local and out of git:

- `data/kitti_raw/` for KITTI Raw
- `data/kitti_tracking/` for KITTI Tracking
- `data/cache/` for generated loader caches such as processed `.npz` streams

Those paths are ignored in `.gitignore`. Do not commit KITTI archives, extracted
sequences, generated dataset indexes, processed `.npz` streams, or local replay caches.

The next Stage 1 slice is `prototypes/python/kitti_loader.py`: load KITTI Raw OXTS
with `pykitti`, convert first-frame-origin WGS84 lat/lon/alt to ENU, and cache
processed arrays under `data/cache/`. Stage 0 tests must remain runnable without
KITTI data.

## Linear CV Kalman filter

![](./docs/images/linear_kf_trajectory.png)

## Radar EKF (synthetic)

Trajectory, position error, innovations, and NIS from `prototypes/python/ekf_synthetic.py`.

![](./docs/images/ekf_synthetic_summary.png)

## Radar UKF (synthetic)

Same scenario as EKF; Julier-scaled sigma points (α=1e-3, β=2, κ=0). EKF overlay on trajectory/error panels for comparison.

![](./docs/images/ukf_synthetic_summary.png)

## IMM CV / CA / CT (synthetic)

Maneuvering target (CV → coordinated turn → CV) with IMM mixing CV-KF, CA-KF, and CT-UKF; position measurements.

![](./docs/images/imm_synthetic_summary.png)
