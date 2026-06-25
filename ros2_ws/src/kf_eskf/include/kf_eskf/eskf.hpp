// 15-state Error-State EKF (ESKF) for IMU + GPS-position fusion.
//
// A 1:1 C++ port of prototypes/python/eskf.py (the ErrorStateEKF class) and the
// nominal strapdown from prototypes/python/imu_mechanization.py. ROS-independent:
// only Eigen and kf_common::so3 are used, so this header host-compiles and is the
// math kernel a later ROS2 node will wrap.
//
// Error-state order: [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)] = 15.
// GLOBAL (world/map-frame) angular-error convention (Sola): q_true = Exp(dtheta) * q_nom.
// Quaternions are scalar-first (Eigen::Quaterniond, w first) and rotate IMU body -> map (ENU).
#ifndef KF_ESKF_ESKF_HPP
#define KF_ESKF_ESKF_HPP

#include <Eigen/Dense>

#include "kf_common/so3.hpp"

namespace kf_eskf {

inline constexpr int kStateDim = 15;  // dp + dv + dtheta + db_a + db_g
inline constexpr int kGpsDim = 3;     // ENU position measurement

// ENU gravity, z-up so z is negative. Mirrors imu_mechanization.GRAVITY_ENU.
inline const Eigen::Vector3d kGravityEnu(0.0, 0.0, -9.80665);

using Matrix15d = Eigen::Matrix<double, kStateDim, kStateDim>;
using Vector15d = Eigen::Matrix<double, kStateDim, 1>;

// Configuration + noise model. Same fields/defaults as the Python EskfConfig.
struct EskfConfig {
  double sigma_accel = 2.0;          // m/s^2, accel white noise
  double sigma_gyro = 1e-3;          // rad/s, gyro white noise
  double sigma_accel_bias = 1e-3;    // m/s^2 random walk
  double sigma_gyro_bias = 1e-5;     // rad/s random walk
  double gps_std_m = 0.75;           // GPS position noise std (also injected noise std)
  int gps_rate_divisor = 10;         // GPS every Nth IMU sample
  double p0_pos = 2.0;               // initial position std, m
  double p0_vel = 3.0;               // initial velocity std, m/s
  double p0_att = 0.5 * M_PI / 180.0;  // initial attitude std, rad (deg2rad(0.5))
  double p0_accel_bias = 0.1;        // initial accel-bias std, m/s^2
  double p0_gyro_bias = 1e-4;        // initial gyro-bias std, rad/s
  Eigen::Vector3d lever = Eigen::Vector3d::Zero();  // GPS antenna lever arm in body frame

  // Initial 15x15 covariance: diag of per-block variances. Mirrors initial_covariance().
  Matrix15d initialCovariance() const {
    Vector15d diag;
    diag.segment<3>(0).setConstant(p0_pos * p0_pos);
    diag.segment<3>(3).setConstant(p0_vel * p0_vel);
    diag.segment<3>(6).setConstant(p0_att * p0_att);
    diag.segment<3>(9).setConstant(p0_accel_bias * p0_accel_bias);
    diag.segment<3>(12).setConstant(p0_gyro_bias * p0_gyro_bias);
    return diag.asDiagonal();
  }

  // GPS measurement covariance R = gps_std^2 * I. Mirrors gps_covariance().
  Eigen::Matrix3d gpsCovariance() const {
    return (gps_std_m * gps_std_m) * Eigen::Matrix3d::Identity();
  }

  // 15x15 process noise Q added each predict. Python process_noise() returns a 12x12
  // that predict() maps through Fi into the 15-state; we build the 15x15 directly with
  // the same dt^2 (white noise) / dt (bias random walk) scaling. Position rows get none.
  Matrix15d processNoise(double dt) const {
    Vector15d diag = Vector15d::Zero();
    diag.segment<3>(3).setConstant(sigma_accel * sigma_accel * dt * dt);        // accel white -> velocity
    diag.segment<3>(6).setConstant(sigma_gyro * sigma_gyro * dt * dt);          // gyro white -> attitude
    diag.segment<3>(9).setConstant(sigma_accel_bias * sigma_accel_bias * dt);   // accel bias random walk
    diag.segment<3>(12).setConstant(sigma_gyro_bias * sigma_gyro_bias * dt);    // gyro bias random walk
    return diag.asDiagonal();
  }
};

// 15-state error-state EKF: IMU predict + GPS-position update.
class ErrorStateEkf {
 public:
  ErrorStateEkf(const Eigen::Vector3d& position,
                const Eigen::Vector3d& velocity,
                const Eigen::Quaterniond& q_map_imu,
                const EskfConfig& config,
                const Eigen::Vector3d& accel_bias = Eigen::Vector3d::Zero(),
                const Eigen::Vector3d& gyro_bias = Eigen::Vector3d::Zero())
      : position_(position),
        velocity_(velocity),
        q_map_imu_(kf_common::normalize(q_map_imu)),
        accel_bias_(accel_bias),
        gyro_bias_(gyro_bias),
        config_(config),
        P_(config.initialCovariance()) {}

  // IMU predict: propagate the error-state covariance, then the nominal strapdown.
  void predict(const Eigen::Vector3d& accel_body, const Eigen::Vector3d& gyro_body, double dt) {
    const Matrix15d fx = stateTransition(accel_body, dt);
    const Matrix15d q = config_.processNoise(dt);
    // P <- Fx P Fx^T + Q. Fi just scatters the 12 noise terms onto rows 3..15, which is
    // exactly how processNoise() already places them, so no explicit Fi is needed here.
    P_ = fx * P_ * fx.transpose() + q;
    P_ = 0.5 * (P_ + P_.transpose().eval());  // re-symmetrize: kill numerical drift
    propagateNominal(accel_body, gyro_body, dt);
  }

  // GPS position update: innovation, Joseph-form covariance, then inject + reset.
  void updateGps(const Eigen::Vector3d& z_enu) {
    const Eigen::Matrix3d rotation = kf_common::toRotationMatrix(q_map_imu_);
    Eigen::Matrix<double, kGpsDim, kStateDim> h = Eigen::Matrix<double, kGpsDim, kStateDim>::Zero();
    h.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    h.block<3, 3>(0, 6) = -kf_common::skew(rotation * config_.lever);  // dh/d(dtheta), global form

    const Eigen::Vector3d y = z_enu - (position_ + rotation * config_.lever);  // innovation
    const Eigen::Matrix3d r_gps = config_.gpsCovariance();
    const Eigen::Matrix3d s = h * P_ * h.transpose() + r_gps;  // innovation covariance
    // K = P H^T S^-1, solved (never an explicit inverse): K^T = S^-1 (H P).
    const Eigen::Matrix<double, kStateDim, kGpsDim> gain =
        s.ldlt().solve(h * P_).transpose();
    const Vector15d dx = gain * y;

    // Joseph form: stays PSD even with a non-optimal gain.
    const Matrix15d i_kh = Matrix15d::Identity() - gain * h;
    P_ = i_kh * P_ * i_kh.transpose() + gain * r_gps * gain.transpose();
    P_ = 0.5 * (P_ + P_.transpose().eval());
    injectAndReset(dx);
  }

  // Accessors.
  const Eigen::Vector3d& position() const { return position_; }
  const Eigen::Vector3d& velocity() const { return velocity_; }
  const Eigen::Quaterniond& quaternion() const { return q_map_imu_; }
  const Eigen::Vector3d& accelBias() const { return accel_bias_; }
  const Eigen::Vector3d& gyroBias() const { return gyro_bias_; }
  const Matrix15d& covariance() const { return P_; }

  // Error-state transition Jacobian Fx, GLOBAL angular-error convention. Public so
  // tests can compare it against finite differences. Mirrors _state_transition().
  Matrix15d stateTransition(const Eigen::Vector3d& accel_body, double dt) const {
    const Eigen::Matrix3d rotation = kf_common::toRotationMatrix(q_map_imu_);
    const Eigen::Vector3d rotated_accel = rotation * (accel_body - accel_bias_);  // R (a - b_a)
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

    Matrix15d fx = Matrix15d::Identity();
    fx.block<3, 3>(0, 3) = identity * dt;
    fx.block<3, 3>(0, 6) = -0.5 * kf_common::skew(rotated_accel) * dt * dt;
    fx.block<3, 3>(0, 9) = -0.5 * rotation * dt * dt;
    fx.block<3, 3>(3, 6) = -kf_common::skew(rotated_accel) * dt;
    fx.block<3, 3>(3, 9) = -rotation * dt;
    fx.block<3, 3>(6, 12) = -rotation * dt;
    return fx;
  }

 private:
  // Nominal strapdown, a port of imu_mechanization.propagate_state. LOCAL gyro
  // integration: delta_q = Exp((gyro - b_g) dt) and q_new = q * delta_q (right multiply,
  // body rotates in body frame). Position uses the 0.5 dt^2 second-order accel term.
  void propagateNominal(const Eigen::Vector3d& accel_body, const Eigen::Vector3d& gyro_body, double dt) {
    const Eigen::Matrix3d rotation = kf_common::toRotationMatrix(q_map_imu_);
    const Eigen::Vector3d accel = accel_body - accel_bias_;
    const Eigen::Vector3d gyro = gyro_body - gyro_bias_;
    const Eigen::Vector3d accel_map = rotation * accel + kGravityEnu;

    position_ = position_ + velocity_ * dt + 0.5 * accel_map * dt * dt;
    velocity_ = velocity_ + accel_map * dt;
    const Eigen::Quaterniond delta_q = kf_common::expMap(gyro * dt);
    q_map_imu_ = kf_common::normalize(q_map_imu_ * delta_q);  // right-multiply, matches Python
  }

  // Inject dx into the nominal state via boxplus, then reset the error to zero. The
  // reset rotates the attitude frame, so P is rotated by G to stay consistent.
  void injectAndReset(const Vector15d& dx) {
    const Eigen::Vector3d dp = dx.segment<3>(0);
    const Eigen::Vector3d dv = dx.segment<3>(3);
    const Eigen::Vector3d dtheta = dx.segment<3>(6);
    const Eigen::Vector3d dba = dx.segment<3>(9);
    const Eigen::Vector3d dbg = dx.segment<3>(12);

    position_ += dp;
    velocity_ += dv;
    q_map_imu_ = kf_common::boxplus(q_map_imu_, dtheta);  // global left-multiply Exp(dtheta) * q
    accel_bias_ += dba;
    gyro_bias_ += dbg;

    Matrix15d g = Matrix15d::Identity();
    g.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() - kf_common::skew(0.5 * dtheta);  // reset Jacobian
    P_ = g * P_ * g.transpose();
    P_ = 0.5 * (P_ + P_.transpose().eval());
  }

  Eigen::Vector3d position_;        // ENU position, m
  Eigen::Vector3d velocity_;        // ENU velocity, m/s
  Eigen::Quaterniond q_map_imu_;    // scalar-first, rotates IMU body -> map (ENU)
  Eigen::Vector3d accel_bias_;      // current accel-bias estimate, m/s^2
  Eigen::Vector3d gyro_bias_;       // current gyro-bias estimate, rad/s
  EskfConfig config_;
  Matrix15d P_;                     // 15x15 error-state covariance
};

}  // namespace kf_eskf

#endif  // KF_ESKF_ESKF_HPP
