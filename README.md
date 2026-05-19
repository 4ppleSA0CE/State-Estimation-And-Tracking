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