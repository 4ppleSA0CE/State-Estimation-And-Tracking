# Porting the Error-State EKF to C++ as a ROS2 Node

The validated Python ESKF (loosely-coupled IMU + GPS INS on KITTI) moved into a
production-grade C++ ROS2 node, validated bit-for-bit against the prototype.

## Architecture

Four packages in `ros2_ws/src/`, built with `colcon` inside a ROS2 Humble Docker
container (`docker/Dockerfile`, `docker/docker-compose.yml`):

- **`kf_common`** — header-only scalar-first SO(3)/quaternion math (exp/log maps,
  skew, boxplus/boxminus) in Eigen. A 1:1 mirror of `prototypes/python/so3.py`.
- **`kf_msgs`** — the custom interfaces: `EgoState`, `Detection`, `DetectionArray`,
  `Track`, `TrackArray`.
- **`kf_eskf`** — `kf_eskf::ErrorStateEkf` (the filter, ported from `eskf.py`) plus
  `EskfNode` (the ROS wrapper).
- **`kf_bringup`** — the `kitti_replay` node, launch file, and YAML config.

Data flow: `kitti_replay` publishes `/imu/data` (100 Hz) and `/gps/fix` (10 Hz) from
the cached KITTI OXTS; `EskfNode` predicts on IMU, updates on GPS, publishes
`/ego/state` (`kf_msgs/EgoState`) and broadcasts the `map -> base_link` TF.

## Key decisions

**Callback-per-sensor, not a synchronizer.** The filter is loosely coupled: IMU
drives predict, GPS drives update, and the two are independent events. A
`message_filters` approximate-time synchronizer would force both onto a single
callback and add latency for no benefit. Instead each sensor has its own callback
on a single-threaded executor, so events are processed in arrival order. The
ordering that matters — predict before update within a step — is guaranteed by the
replay publishing `IMU[k-1]` before `GPS[k]` (see below), not by a synchronizer.

**Hand-rolled SO(3), not Sophus.** The Python prototype uses scalar-first `[w,x,y,z]`
quaternions with a specific global/local convention split (global left-multiply for
the error injection, local right-multiply for the gyro strapdown). Re-deriving that
in `kf_common` against Eigen keeps an exact parity guarantee with the prototype and
avoids a from-source Sophus build with its own quaternion layout. ~150 lines, fully
gtested.

**YAML parameters, not hard-coded tuning.** All noise/init/frame parameters live in
`config/eskf_kitti.yaml`, loaded via ROS params. The tuning is versionable and
reproducible, and the node carries no magic numbers.

**The predict -> update -> record ordering (the parity-critical detail).** The Python
loop is `predict(IMU[k-1]) -> update_gps(GPS[k]) -> record`. To reproduce this over
asynchronous ROS topics: the replay publishes `IMU[k-1]` (stamped `t[k]`) then
`GPS[k]`; the node predicts in the IMU callback and updates in the GPS callback, but
publishes the *previous* completed step at the start of each IMU callback — by which
point that step's predict and any GPS update are done. A final duplicate-stamp IMU
flushes the last step. This yields exactly one `EgoState` per step, post-update, in
order. Getting this wrong (update before predict) cost 0.22 m of error; getting it
right drove parity to zero.

## Validation

`kitti_replay` reproduces the prototype's seeded GPS noise exactly (same
`np.random.default_rng(seed)` call order as `build_gps_measurements`), so the C++
node receives byte-identical measurements. Against the Python reference trajectory
(`run_eskf`, seed 0) over the full 1166-sample drive:

```
PARITY position_rmse_m=0.0000  max_abs_m=0.0000  n=1165   PASS
```

The C++ node matches the prototype to numerical precision — far inside the 0.1 m
DOD. This is the payoff of mirroring the math 1:1 and feeding identical inputs.

## Performance

- **Predict + (1-in-10) GPS update: ~4.6 µs/step** (`-O2`, measured over 200k steps,
  ~217 kHz capable). At the 100 Hz IMU rate that is ~0.05% of one core, so the filter
  is effectively free; headroom is enormous for higher rates or extra sensors.
- **End-to-end:** the 11.6 s drive (1166 samples) replays and filters in ~1 s wall
  clock, throttled only by the replay's 1 ms inter-message pacing that keeps the
  reliable queue drained.

## Tests and CI

- `kf_common`: 6 gtests (skew = cross product, exp/log round-trip incl. small-angle,
  boxplus/boxminus manifold round-trip).
- `kf_eskf`: 4 gtests (predict Jacobian vs finite differences at 5e-6, covariance
  stays symmetric-PSD, GPS update pulls toward the measurement, stationary
  convergence).
- `.github/workflows/ci.yml` builds + runs both the Python pytest suite and the C++
  colcon/gtest suite in a ROS2 Humble container on every push and PR.
