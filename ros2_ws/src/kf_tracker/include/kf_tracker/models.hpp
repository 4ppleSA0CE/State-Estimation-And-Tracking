// IMM per-mode motion filters: CV (linear KF), CA (6-state linear KF), CT (coordinated-turn UKF).
//
// A 1:1 C++ port of the *ModelFilter classes in prototypes/python/imm_synthetic.py, plus
// cv_matrices/LinearKF from prototypes/python/linear_kf.py. ROS-independent — Eigen and
// kf_common::unscented only, so this header host-compiles.
//
// The modes do NOT share a state dimension. They share a 4-D REFERENCE space [x, y, vx, vy] in
// which IMM mixing happens; setState() inflates from it and refState() deflates back to it. For
// CA those two must be exact inverses — a permutation bug there is invisible to a parity test
// where both sides call the same converter (see docs/notes/tracking_imm_writeup.md).
//
// Linear modes use the Joseph form P = (I-KH) P (I-KH)^T + K R K^T, matching LinearKF.
#ifndef KF_TRACKER_MODELS_HPP
#define KF_TRACKER_MODELS_HPP

#include <cmath>

#include <Eigen/Dense>

#include "kf_common/unscented.hpp"

namespace kf_tracker {

inline constexpr int kRefDim = 4;    // [x, y, vx, vy] mixing space
inline constexpr int kMeasDim = 2;   // position-only measurement
inline constexpr int kCaDim = 6;     // [x, y, vx, vy, ax, ay]

using Vector4d = Eigen::Matrix<double, kRefDim, 1>;
using Matrix4d = Eigen::Matrix<double, kRefDim, kRefDim>;
using Vector2d = Eigen::Matrix<double, kMeasDim, 1>;
using Matrix2d = Eigen::Matrix<double, kMeasDim, kMeasDim>;
using Vector6d = Eigen::Matrix<double, kCaDim, 1>;
using Matrix6d = Eigen::Matrix<double, kCaDim, kCaDim>;

// Scalar measurement likelihood N(y; 0, S). Mirrors imm_synthetic.gaussian_likelihood, including
// the non-positive-determinant floor.
//
// The explicit determinant()/inverse() here is deliberate and is NOT the style regression it looks
// like next to kf_eskf/eskf.hpp's s.ldlt().solve(...). S is 2x2, so Eigen uses the closed-form
// adjugate/determinant rather than a factorization, and the floor branch needs the signed
// determinant anyway -- an LDLT rewrite would have to compute one separately and would change the
// rounding of every pinned likelihood in test_models.cpp / test_imm.cpp. Leave it.
inline double gaussianLikelihood(const Vector2d& y, const Matrix2d& s) {
  const double det = s.determinant();
  if (det <= 0.0) return 1e-300;
  // (0,0) is required: the triple product is a 1x1 Eigen expression, not a scalar.
  const double quad = (y.transpose() * s.inverse() * y)(0, 0);
  return std::exp(-0.5 * (kMeasDim * std::log(2.0 * M_PI) + std::log(det) + quad));
}

// Coordinated turn on [x, y, vx, vy] at constant turn rate omega. Mirrors ct_propagate_state,
// including the |omega| < 1e-8 straight-line guard.
inline Vector4d ctPropagate(const Vector4d& x, double dt, double omega) {
  const double px = x(0), py = x(1), vx = x(2), vy = x(3);
  if (std::abs(omega) < 1e-8) {
    Vector4d out;
    out << px + vx * dt, py + vy * dt, vx, vy;
    return out;
  }
  const double s = std::sin(omega * dt);
  const double c = std::cos(omega * dt);
  Vector4d out;
  out << px + (vx * s + vy * (1.0 - c)) / omega,
         py + (vy * s - vx * (1.0 - c)) / omega,
         c * vx - s * vy,
         s * vx + c * vy;
  return out;
}

// Abstract per-mode filter. All I/O is in the 4-D reference space.
class ModelFilter {
 public:
  virtual ~ModelFilter() = default;
  virtual void setState(const Vector4d& x_ref, const Matrix4d& p_ref) = 0;
  virtual void predict() = 0;
  virtual double update(const Vector2d& z) = 0;   // returns the measurement likelihood
  virtual void refState(Vector4d& x_ref, Matrix4d& p_ref) const = 0;
};

// ---------------------------------------------------------------------------
// CV — 4-state constant velocity. Reference space IS the internal state.
//
// MEASUREMENT NOISE: `r` is the ONE source of R here. Python's CvModelFilter accepts an `r`
// argument but immediately overwrites it with cv_matrices' sigma_pos^2 * I, so on that side `r`
// is dead and sigma_pos wins. The two disagree at a real call site:
// prototypes/python/tests/test_imm_filter.py:69 builds IMMFilter(IMMConfig(omegas=(0.25,-0.25)),
// r=np.eye(2)*2.0) against IMMConfig's default sigma_pos=1.0, i.e. r = 2I but sigma_pos^2 I = I.
// That is not a rounding artifact — one predict+update from x=[1,2,3,4], P=diag(1,2,5,7),
// z=[1.35,2.45] gives x(0) 1.3172136522352094 honouring r=2I vs 1.3256103509670496 honouring
// sigma_pos^2, and likelihoods 0.0451 vs 0.0634 (~40% relative), four orders past the 1e-6
// parity gate. It stays latent only because the tracker's own call site happens to pass
// sigma_pos^2 * I.
//
// So the sigma_pos parameter is GONE from CvModelFilter and CaModelFilter rather than sitting
// there unused: with no second noise input the divergence is structurally impossible to
// reintroduce. ImmConfig::sigma_pos survives as the caller's handle for BUILDING r (see
// ImmFilter's constructor in imm.hpp) and no longer reaches these filters. Same for CA below.
// ---------------------------------------------------------------------------
class CvModelFilter : public ModelFilter {
 public:
  CvModelFilter(double dt, double q_accel, const Matrix2d& r) : r_(r) {
    f_ = Matrix4d::Identity();
    f_(0, 2) = dt;
    f_(1, 3) = dt;

    h_.setZero();
    h_(0, 0) = 1.0;
    h_(1, 1) = 1.0;

    // Per-axis white-noise-acceleration blocks on (0,2) and (1,3). Mirrors cv_matrices.
    Eigen::Vector2d g1d;
    g1d << 0.5 * dt * dt, dt;
    const Eigen::Matrix2d q1d = q_accel * (g1d * g1d.transpose());
    q_.setZero();
    q_(0, 0) = q1d(0, 0); q_(0, 2) = q1d(0, 1); q_(2, 0) = q1d(1, 0); q_(2, 2) = q1d(1, 1);
    q_(1, 1) = q1d(0, 0); q_(1, 3) = q1d(0, 1); q_(3, 1) = q1d(1, 0); q_(3, 3) = q1d(1, 1);
  }

  void setState(const Vector4d& x_ref, const Matrix4d& p_ref) override {
    x_ = x_ref;
    p_ = p_ref;
  }

  void predict() override {
    x_ = f_ * x_;
    p_ = f_ * p_ * f_.transpose() + q_;
  }

  double update(const Vector2d& z) override {
    const Vector2d y = z - h_ * x_;
    const Matrix2d s = h_ * p_ * h_.transpose() + r_;
    const Eigen::Matrix<double, kRefDim, kMeasDim> k = p_ * h_.transpose() * s.inverse();
    x_ = x_ + k * y;
    const Matrix4d i_kh = Matrix4d::Identity() - k * h_;
    p_ = i_kh * p_ * i_kh.transpose() + k * r_ * k.transpose();   // Joseph form
    return gaussianLikelihood(y, s);
  }

  void refState(Vector4d& x_ref, Matrix4d& p_ref) const override {
    x_ref = x_;
    p_ref = p_;
  }

 private:
  Matrix4d f_, q_, p_ = Matrix4d::Identity();
  Eigen::Matrix<double, kMeasDim, kRefDim> h_;
  Matrix2d r_;
  Vector4d x_ = Vector4d::Zero();
};

// ---------------------------------------------------------------------------
// CA — 6-state constant acceleration, order [x, y, vx, vy, ax, ay].
// ---------------------------------------------------------------------------
class CaModelFilter : public ModelFilter {
 public:
  CaModelFilter(double dt, double q_accel, const Matrix2d& r) : r_(r) {
    // Per-axis [pos, vel, acc] block on indices {0,2,4} (x) and {1,3,5} (y). Mirrors ca_matrices.
    Eigen::Matrix3d f1;
    f1 << 1.0, dt, 0.5 * dt * dt,
          0.0, 1.0, dt,
          0.0, 0.0, 1.0;
    Eigen::Vector3d g;
    g << 0.5 * dt * dt, dt, 1.0;
    const Eigen::Matrix3d q1 = q_accel * (g * g.transpose());

    f_.setZero();
    q_.setZero();
    const int ix[3] = {0, 2, 4};
    const int iy[3] = {1, 3, 5};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        f_(ix[a], ix[b]) = f1(a, b);
        f_(iy[a], iy[b]) = f1(a, b);
        q_(ix[a], ix[b]) = q1(a, b);
        q_(iy[a], iy[b]) = q1(a, b);
      }
    }

    h_.setZero();
    h_(0, 0) = 1.0;
    h_(1, 1) = 1.0;
  }

  // Inflate: seed the accel block, then overwrite the leading 4x4 with the reference block.
  // Mirrors ref_to_ca AFTER its 2026-07-27 fix — a PLAIN block copy, no index permutation.
  void setState(const Vector4d& x_ref, const Matrix4d& p_ref) override {
    x_.setZero();
    x_.head<kRefDim>() = x_ref;
    p_.setZero();
    p_.diagonal() << 10.0, 10.0, 10.0, 10.0, 5.0, 5.0;
    p_.topLeftCorner<kRefDim, kRefDim>() = p_ref;
  }

  void predict() override {
    x_ = f_ * x_;
    p_ = f_ * p_ * f_.transpose() + q_;
  }

  double update(const Vector2d& z) override {
    const Vector2d y = z - h_ * x_;
    const Matrix2d s = h_ * p_ * h_.transpose() + r_;
    const Eigen::Matrix<double, kCaDim, kMeasDim> k = p_ * h_.transpose() * s.inverse();
    x_ = x_ + k * y;
    const Matrix6d i_kh = Matrix6d::Identity() - k * h_;
    p_ = i_kh * p_ * i_kh.transpose() + k * r_ * k.transpose();   // Joseph form
    return gaussianLikelihood(y, s);
  }

  // Deflate: the exact inverse of setState's leading-block copy. Mirrors ref_from_ca.
  void refState(Vector4d& x_ref, Matrix4d& p_ref) const override {
    x_ref = x_.head<kRefDim>();
    p_ref = p_.topLeftCorner<kRefDim, kRefDim>();
  }

 private:
  Matrix6d f_, q_, p_ = Matrix6d::Identity();
  Eigen::Matrix<double, kMeasDim, kCaDim> h_;
  Matrix2d r_;
  Vector6d x_ = Vector6d::Zero();
};

// ---------------------------------------------------------------------------
// CT — 4-state coordinated turn, propagated by the UNSCENTED transform (not a Jacobian).
// ---------------------------------------------------------------------------
class CtModelFilter : public ModelFilter {
 public:
  CtModelFilter(double dt, double omega, double q_accel, const Matrix2d& r)
      : dt_(dt), omega_(omega), r_(r) {
    Eigen::Vector2d g1d;
    g1d << 0.5 * dt * dt, dt;
    const Eigen::Matrix2d q1d = q_accel * (g1d * g1d.transpose());
    q_.setZero();
    q_(0, 0) = q1d(0, 0); q_(0, 2) = q1d(0, 1); q_(2, 0) = q1d(1, 0); q_(2, 2) = q1d(1, 1);
    q_(1, 1) = q1d(0, 0); q_(1, 3) = q1d(0, 1); q_(3, 1) = q1d(1, 0); q_(3, 3) = q1d(1, 1);

    weights_ = kf_common::unscentedWeights(kRefDim);
  }

  void setState(const Vector4d& x_ref, const Matrix4d& p_ref) override {
    x_ = x_ref;
    p_ = p_ref;
  }

  // sigmaPoints() throws std::runtime_error if P is not PD even after its 1e-9 jitter retry
  // (Python raises LinAlgError there). Deliberately NOT caught: policy lives at the ROS callback.
  void predict() override {
    const Eigen::MatrixXd chi = kf_common::sigmaPoints(x_, p_, weights_.lambda);
    Eigen::MatrixXd chi_pred(kRefDim, chi.cols());
    for (int i = 0; i < chi.cols(); ++i) chi_pred.col(i) = ctPropagate(chi.col(i), dt_, omega_);

    const Eigen::VectorXd mean = kf_common::weightedMean(chi_pred, weights_.wm);
    // Named local: weightedCovariance takes a POINTER, so &temporary would dangle.
    const Eigen::MatrixXd q_dyn = q_;
    p_ = kf_common::weightedCovariance(chi_pred, mean, weights_.wc, &q_dyn);
    x_ = mean;
  }

  double update(const Vector2d& z) override {
    const Eigen::MatrixXd chi = kf_common::sigmaPoints(x_, p_, weights_.lambda);

    Eigen::Matrix<double, kMeasDim, kRefDim> h;
    h.setZero();
    h(0, 0) = 1.0;
    h(1, 1) = 1.0;

    const Eigen::MatrixXd z_sigmas = h * chi;
    const Eigen::VectorXd z_pred = kf_common::weightedMean(z_sigmas, weights_.wm);
    const Vector2d y = z - z_pred;

    const Eigen::MatrixXd r_dyn = r_;
    const Eigen::MatrixXd p_zz =
        kf_common::weightedCovariance(z_sigmas, z_pred, weights_.wc, &r_dyn);

    const Eigen::MatrixXd diff_z = z_sigmas.colwise() - z_pred;
    // Centre on the STORED prior mean x_, exactly as Python's `(chi - self.x)` does — NOT on a
    // recomputed weightedMean(chi, wm). Keep it that way for form-for-form parity, but do not
    // expect a test to catch a swap: the two centres differ by ~2e-10 (the unscented weights only
    // sum to 1 to 4.4e-11, see test_models.cpp), and because ctPropagate is an exactly LINEAR map
    // the first-order effect on p_xz cancels. Measured max|p_xz difference| is 2.2e-16 against a
    // max|p_xz| of 2.07 — one ULP. Only a state-dependent turn rate would make this bite.
    const Eigen::MatrixXd diff_x = chi.colwise() - x_;
    const Eigen::MatrixXd p_xz = diff_x * weights_.wc.asDiagonal() * diff_z.transpose();

    const Eigen::MatrixXd k = p_xz * p_zz.inverse();
    x_ = x_ + k * y;
    p_ = p_ - k * p_zz * k.transpose();
    // .eval() is required because this is a coefficient-wise assignment that reads p_(j,i) while
    // writing p_(i,j): without the temporary, column-major traversal has already overwritten the
    // lower triangle by the time the upper triangle reads it, leaving 0.75a + 0.25b there against
    // 0.5(a+b) below. It is harmless at THIS call site only because p_ is already symmetric to
    // roundoff by the time we arrive, so the aliased read costs ~1 ULP; the .eval() is what keeps
    // the symmetry invariant from depending on that. Feed it a genuinely non-symmetric p_ and the
    // difference is O(asymmetry/4) — see CtSymmetrizeIsAliasFree in test_models.cpp.
    p_ = 0.5 * (p_ + p_.transpose()).eval() + 1e-12 * Matrix4d::Identity();   // symmetrize

    return gaussianLikelihood(y, p_zz);
  }

  void refState(Vector4d& x_ref, Matrix4d& p_ref) const override {
    x_ref = x_;
    p_ref = p_;
  }

 private:
  double dt_;
  double omega_;
  Matrix4d q_, p_ = Matrix4d::Identity();
  Matrix2d r_;
  Vector4d x_ = Vector4d::Zero();
  kf_common::UnscentedWeights weights_;
};

}  // namespace kf_tracker

#endif  // KF_TRACKER_MODELS_HPP
