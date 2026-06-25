// gtest for the 15-state ESKF. Mirrors the key tests in prototypes/python/tests/test_eskf.py
// so the C++ and Python filters stay in lockstep (later validated to <0.1 m RMSE).
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "kf_common/so3.hpp"
#include "kf_eskf/eskf.hpp"

using kf_eskf::ErrorStateEkf;
using kf_eskf::EskfConfig;
using kf_eskf::Matrix15d;
using kf_eskf::Vector15d;
using kf_eskf::kGravityEnu;
using kf_eskf::kStateDim;

namespace {

// Smallest eigenvalue of the symmetric part of P; a proxy for "stays PSD".
double minEigenvalue(const Matrix15d& p) {
  const Matrix15d sym = 0.5 * (p + p.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix15d> solver(sym);
  return solver.eigenvalues().minCoeff();
}

// boxplus on the full 15-state nominal: vector parts add, attitude composes globally.
struct Nominal {
  Eigen::Vector3d position;
  Eigen::Vector3d velocity;
  Eigen::Quaterniond q;
  Eigen::Vector3d accel_bias;
  Eigen::Vector3d gyro_bias;
};

Nominal boxplusNominal(const Nominal& n, const Vector15d& dx) {
  Nominal out;
  out.position = n.position + dx.segment<3>(0);
  out.velocity = n.velocity + dx.segment<3>(3);
  out.q = kf_common::boxplus(n.q, dx.segment<3>(6));  // Exp(dtheta) * q, global
  out.accel_bias = n.accel_bias + dx.segment<3>(9);
  out.gyro_bias = n.gyro_bias + dx.segment<3>(12);
  return out;
}

// boxminus: the 15-vector error such that a == boxplus(b, dx).
Vector15d boxminusNominal(const Nominal& a, const Nominal& b) {
  Vector15d dx;
  dx.segment<3>(0) = a.position - b.position;
  dx.segment<3>(3) = a.velocity - b.velocity;
  dx.segment<3>(6) = kf_common::boxminus(a.q, b.q);
  dx.segment<3>(9) = a.accel_bias - b.accel_bias;
  dx.segment<3>(12) = a.gyro_bias - b.gyro_bias;
  return dx;
}

// Pure nominal propagation (no covariance), matching ErrorStateEkf::propagateNominal,
// used to build the numerical Jacobian the analytic Fx is compared against.
Nominal propagateNominal(const Nominal& n, const Eigen::Vector3d& accel_body,
                         const Eigen::Vector3d& gyro_body, double dt) {
  const Eigen::Matrix3d rotation = kf_common::toRotationMatrix(n.q);
  const Eigen::Vector3d accel = accel_body - n.accel_bias;
  const Eigen::Vector3d gyro = gyro_body - n.gyro_bias;
  const Eigen::Vector3d accel_map = rotation * accel + kGravityEnu;

  Nominal out;
  out.position = n.position + n.velocity * dt + 0.5 * accel_map * dt * dt;
  out.velocity = n.velocity + accel_map * dt;
  out.q = kf_common::normalize(n.q * kf_common::expMap(gyro * dt));
  out.accel_bias = n.accel_bias;
  out.gyro_bias = n.gyro_bias;
  return out;
}

// The predict Jacobian must match a central finite-difference of the nominal flow.
TEST(Eskf, PredictJacobianMatchesFiniteDifference) {
  Nominal n;
  n.position = Eigen::Vector3d(1.0, -2.0, 0.5);
  n.velocity = Eigen::Vector3d(0.3, 0.1, -0.2);
  n.q = kf_common::expMap(Eigen::Vector3d(0.1, -0.2, 0.05));
  n.accel_bias = Eigen::Vector3d(0.02, -0.01, 0.03);
  n.gyro_bias = Eigen::Vector3d(-0.001, 0.002, 0.0);

  const Eigen::Vector3d accel_body(0.5, -0.3, 9.9);
  const Eigen::Vector3d gyro_body(0.02, -0.05, 0.1);
  const double dt = 0.01;

  ErrorStateEkf filt(n.position, n.velocity, n.q, EskfConfig(), n.accel_bias, n.gyro_bias);
  const Matrix15d fx = filt.stateTransition(accel_body, dt);

  const double eps = 1e-6;
  Matrix15d numeric;
  for (int j = 0; j < kStateDim; ++j) {
    Vector15d perturb = Vector15d::Zero();
    perturb(j) = eps;
    const Nominal plus = propagateNominal(boxplusNominal(n, perturb), accel_body, gyro_body, dt);
    perturb(j) = -eps;
    const Nominal minus = propagateNominal(boxplusNominal(n, perturb), accel_body, gyro_body, dt);
    numeric.col(j) = boxminusNominal(plus, minus) / (2.0 * eps);
  }
  EXPECT_LT((fx - numeric).cwiseAbs().maxCoeff(), 1e-5);
}

// Covariance stays symmetric and PSD after 50 predicts.
TEST(Eskf, CovarianceStaysSymmetricPsd) {
  ErrorStateEkf filt(Eigen::Vector3d::Zero(), Eigen::Vector3d(5.0, 0.0, 0.0),
                     Eigen::Quaterniond::Identity(), EskfConfig());
  const Eigen::Vector3d accel_body(0.0, 0.0, -kGravityEnu.z());  // (0,0,+g): hover specific force
  const Eigen::Vector3d gyro_body(0.01, -0.02, 0.005);
  for (int k = 0; k < 50; ++k) {
    filt.predict(accel_body, gyro_body, 0.01);
  }
  const Matrix15d& p = filt.covariance();
  EXPECT_LT((p - p.transpose()).cwiseAbs().maxCoeff(), 1e-9);
  EXPECT_GT(minEigenvalue(p), -1e-9);
}

// A GPS update pulls the nominal position toward the measurement; P stays PSD.
TEST(Eskf, GpsUpdateMovesPositionTowardMeasurement) {
  ErrorStateEkf filt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                     Eigen::Quaterniond::Identity(), EskfConfig());
  const Eigen::Vector3d z(3.0, -1.0, 0.5);
  const double before = (filt.position() - z).norm();
  filt.updateGps(z);
  const double after = (filt.position() - z).norm();
  EXPECT_LT(after, before);
  EXPECT_GT(minEigenvalue(filt.covariance()), -1e-9);
}

// Stationary convergence: identity attitude, specific force (0,0,+g) so the nominal
// hovers, GPS every 10th step at a fixed truth + small offset. Position error < 1.0 m.
TEST(Eskf, StationaryConvergence) {
  EskfConfig config;
  ErrorStateEkf filt(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                     Eigen::Quaterniond::Identity(), config);
  const Eigen::Vector3d accel_body(0.0, 0.0, -kGravityEnu.z());  // (0,0,+9.80665)
  const Eigen::Vector3d gyro_body = Eigen::Vector3d::Zero();
  const Eigen::Vector3d truth(2.0, -1.0, 0.5);
  const Eigen::Vector3d z = truth + Eigen::Vector3d(0.05, -0.05, 0.02);  // fixed small offset

  for (int k = 1; k <= 2000; ++k) {
    filt.predict(accel_body, gyro_body, 0.01);
    if (k % config.gps_rate_divisor == 0) {
      filt.updateGps(z);
    }
  }
  EXPECT_LT((filt.position() - truth).norm(), 1.0);
  EXPECT_GT(minEigenvalue(filt.covariance()), -1e-9);
}

}  // namespace
