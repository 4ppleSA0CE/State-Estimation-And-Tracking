# UKF vs ESKF on KITTI

Both filters share identical noise/init/GPS tuning (`UkfConfig` embeds `EskfConfig`),
100 Hz IMU predict, 10 Hz GPS-position update.

| metric | ESKF | UKF |
| --- | --- | --- |
| position RMSE [m] | 0.410 | 0.410 |
| attitude RMSE [deg] | 0.512 | 0.512 |
| runtime [s] | 0.223 | 5.412 |

UKF runtime is 24.3x the ESKF (it propagates 31 sigma
points through the strapdown per step vs one analytic Jacobian).

## Verdict

On this near-linear INS problem the ESKF and UKF reach effectively the same
accuracy; the ESKF is preferred because it costs a single analytic Jacobian per
step instead of 31 strapdown propagations, so it is far cheaper for no accuracy
loss. The UKF would only pull ahead under strong nonlinearity (large attitude
errors, coarse update rates) that this 100 Hz / 10 Hz setup does not exhibit.

## GPS dropout

Dropout window (4.0, 8.0) s. Position RMSE with the cut:
ESKF 1.508 m, UKF 1.507 m. Both drift
open-loop through the gap on IMU alone and re-converge once GPS returns.
