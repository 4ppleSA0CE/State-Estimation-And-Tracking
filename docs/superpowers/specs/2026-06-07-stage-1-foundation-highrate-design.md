# Stage 1 Foundation and High-Rate OXTS Design

## Context

Stage 1 moves the project from synthetic filters to an Error-State EKF on KITTI data in Python. Stage 1.1 is functionally implemented: `prototypes/python/kitti_loader.py` loads KITTI Raw synced OXTS, converts first-fix WGS84 positions to local ENU, writes a deterministic cache, and has passing tests. The PRD metadata is stale, however, and still reports Stage 1.1 as not started.

Local KITTI data currently contains `*_sync` folders only. Those synced OXTS samples are about 10 Hz and are useful for loader validation, frame smoke tests, low-rate GPS/pose references, and trajectory plots. They are not sufficient for the INS mechanization evidence expected by Stage 1.3 and Stage 1.4. Stage 1 must require high-rate KITTI OXTS from `*_extract` data before claiming mechanization or ESKF completion.

## Approved Direction

Use a foundation-first Stage 1 path:

1. Reconcile PRD status before new Stage 1.2 code.
2. Build Stage 1.2 frame plumbing as a narrow, testable module.
3. Add high-rate OXTS detection/parsing before Stage 1.3 mechanization.
4. Build IMU mechanization, ESKF, UKF, and Stage 1 writeup on stable data and frame contracts.

This avoids building the filter around the current 10 Hz synced OXTS stream and then rewriting it when high-rate data arrives.

## Scope

### In Scope

- Update `PROJECT_PRD.md` progress metadata to show Stage 1.1 functionally done and Stage 1.2 active.
- Add a frame/calibration foundation for `map`, `base_link`, `imu_link`, `gps_link`, and `velo_link`.
- Treat `map` as ENU with origin at the first valid GPS fix.
- Treat `base_link`, `imu_link`, and `gps_link` as colocated by default for Stage 1.
- Make GPS/IMU lever arms configurable and test nonzero offsets.
- Parse KITTI date-level `calib_imu_to_velo.txt` and expose `imu_link <-> velo_link`.
- Detect missing high-rate `*_extract` OXTS data and block Stage 1.3 with a clear setup error.
- Keep Stage 1 Python-only unless a later filter-math parity reason justifies MATLAB.

### Out Of Scope

- Camera intrinsics, rectified camera frames, and camera projection handling.
- ROS2 packages and custom messages.
- C++ ESKF/UKF porting.
- Tracking data and IMM tracker work.
- Inventing unsupported GPS antenna or vehicle-center lever arms.

## Architecture

`kitti_loader.py` remains the synced OXTS loader and cache module. Its current aggregate-array `KittiSequence` API should stay stable unless a cache version bump is needed.

`frames.py` owns frame names, rigid transforms, transform composition/inversion, point/vector/covariance transforms, KITTI calibration parsing, and a `Frames` class that resolves supported transform paths.

`so3.py` or `geometry.py` should be added if reusable rotation/quaternion helpers are needed. Stage 1 should use scalar-first quaternions consistently.

`kitti_highrate_loader.py` should be a separate high-rate OXTS module. It should locate `*_extract/oxts`, parse timestamped high-rate IMU/GPS/OXTS packets, and cache them separately from the synced loader.

`imu_mechanization.py` consumes high-rate IMU samples and frame transforms to propagate nominal position, velocity, and attitude.

`eskf.py` owns the 15-state error-state filter core: predict, GPS update, optional body-velocity update, error injection, covariance reset, and diagnostics.

`ukf_kitti.py` mirrors ESKF I/O for comparison after ESKF data and frame contracts are stable.

Plotting, metric computation, and script entrypoints should stay separate from core filter classes.

## Frame Conventions

- `map`: local ENU world frame, origin at first valid GPS fix.
- `base_link`: vehicle body frame, FLU convention.
- `imu_link`: KITTI OXTS/IMU frame. Stage 1 default is colocated with `base_link`.
- `gps_link`: GPS measurement point. Stage 1 default is colocated with `base_link`.
- `velo_link`: Velodyne frame from KITTI `calib_imu_to_velo.txt`.

Transform notation should be explicit: `T_target_source` maps coordinates from `source` into `target`. Frame-tagged data should reject incompatible source/target use instead of silently returning numeric arrays.

Because KITTI calibration does not provide a project-ready `base_link -> gps_link` or vehicle-center offset in the local files, Stage 1 defaults to identity lever arms while keeping configurable `p_base_gps` and `p_base_imu` offsets.

## Data Flow

The synced loader remains:

```text
data/kitti_raw/..._sync -> KittiSequence -> ENU cache/plot -> validation and low-rate references
```

Stage 1.2 uses:

```text
KittiSequence.origin_lat_lon_alt + date-level calibration -> Frames
```

High-rate prerequisite before Stage 1.3:

```text
data/kitti_raw/<date>/<date>_drive_<drive>_extract/oxts -> high-rate OXTS cache
```

If only `_sync` data exists, the high-rate loader reports the expected `*_extract` path and Stage 1.3 remains blocked.

Mechanization and filters then use:

```text
high-rate IMU -> predict
low-rate GPS position in map -> update
configurable GPS lever arm in base_link -> measurement model
OXTS pose/velocity references -> validation metrics and plots
```

## Error Handling

Frame code should raise clear errors for:

- unknown frame names
- unsupported transform paths
- malformed calibration files
- missing calibration keys
- wrong matrix/vector shapes
- non-orthonormal rotations or determinant not near `+1`
- frame-tag mismatches

High-rate OXTS code should distinguish:

- no high-rate `*_extract` directory
- synced data present but high-rate data absent
- malformed high-rate OXTS timestamps or packets
- cache metadata mismatch

Stage 1.3 must not silently fall back to synced 10 Hz OXTS for mechanization.

## Testing

Primary command:

```bash
venv/bin/python -m pytest prototypes/python/tests -q
```

Stage 1.2 tests should include:

- transform inverse and composition identity
- `map -> base_link -> map` round trips for points
- covariance transform round trips
- vector transforms that ignore translation
- frame-tag mismatch errors
- KITTI-style `R`/`T` fixture parsing
- optional local KITTI calibration integration, skipped when absent
- GPS measurement lever-arm behavior for zero and nonzero offsets
- high-rate loader missing-data error when only `_sync` data exists

Stage 1.3 through 1.5 should add:

- quaternion/SO(3) normalization, exp/log, and composition tests
- stationary IMU mechanization tests
- finite-difference checks for ESKF Jacobians
- covariance symmetry/PSD checks
- deterministic GPS dropout and recovery smoke tests
- ESKF/UKF I/O parity tests on the same data bundle

## Acceptance Criteria

Stage 1.2 is done when:

- `PROJECT_PRD.md` truthfully marks Stage 1.1 done and Stage 1.2 active.
- `frames.py` defines and resolves `map`, `base_link`, `imu_link`, `gps_link`, and `velo_link`.
- KITTI `calib_imu_to_velo.txt` parsing is covered by tests.
- Default colocated GPS/IMU/base behavior is explicit and configurable.
- The high-rate OXTS prerequisite is represented by code or tests that fail clearly when only `_sync` data exists.
- Round-trip transform tests pass.

Stage 1.3 is done when:

- high-rate OXTS data is detected and parsed from `*_extract`
- missing high-rate data produces a clear blocking setup error
- strapdown integration runs on high-rate IMU samples
- short-window drift is measured against OXTS pose

Stage 1.4 is done when:

- ESKF predict uses high-rate IMU
- GPS update uses low-rate map-frame GPS with configurable lever-arm correction
- covariance update/injection/reset behavior is tested
- position, attitude, and consistency metrics are produced

Stage 1.5 is done when:

- UKF consumes the same data bundle and exposes comparable outputs
- ESKF/UKF comparison plots and runtime metrics are generated

Stage 1.6 is done when:

- `docs/stage_notes/stage_1.md` records trajectory plots, error plots, GPS dropout behavior, NEES analysis, and the error-state rationale
- PRD progress metadata reflects Stage 1 completion state

## Implementation Notes

Keep modules small and script-compatible, matching the existing prototype style. Prefer dataclass configs, deterministic tests, `np.linalg.solve` or Cholesky solves over explicit matrix inverses, Joseph-form covariance updates where applicable, and explicit units for IMU/process noise.

Generated caches and plots stay under ignored local paths: `data/cache/` and `prototypes/output/`.
