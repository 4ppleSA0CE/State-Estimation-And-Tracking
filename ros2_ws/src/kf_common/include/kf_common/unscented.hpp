// Unscented transform helpers: weights, sigma points, weighted mean/covariance.
//
// A 1:1 C++ port of prototypes/python/ukf_synthetic.py (unscented_weights, sigma_points,
// weighted_mean, weighted_covariance). ROS-independent — Eigen only. Used by the coordinated-turn
// mode of the IMM tracker (kf_tracker/models.hpp), which is a UKF rather than an EKF.
#ifndef KF_COMMON_UNSCENTED_HPP
#define KF_COMMON_UNSCENTED_HPP

#include <Eigen/Dense>

#include <stdexcept>

namespace kf_common {

// Van der Merwe scaling. Same values as ukf_synthetic.UKF_{ALPHA,BETA,KAPPA}.
inline constexpr double kUkfAlpha = 1e-3;
inline constexpr double kUkfBeta = 2.0;
inline constexpr double kUkfKappa = 0.0;

struct UnscentedWeights {
  double lambda = 0.0;
  Eigen::VectorXd wm;  // mean weights, length 2n+1
  Eigen::VectorXd wc;  // covariance weights, length 2n+1
};

inline UnscentedWeights unscentedWeights(int n,
                                         double alpha = kUkfAlpha,
                                         double beta = kUkfBeta,
                                         double kappa = kUkfKappa) {
  UnscentedWeights w;
  const double dn = static_cast<double>(n);
  w.lambda = alpha * alpha * (dn + kappa) - dn;
  w.wm = Eigen::VectorXd::Constant(2 * n + 1, 1.0 / (2.0 * (dn + w.lambda)));
  w.wc = w.wm;
  w.wm(0) = w.lambda / (dn + w.lambda);
  w.wc(0) = w.lambda / (dn + w.lambda) + (1.0 - alpha * alpha + beta);
  return w;
}

// 2n+1 sigma points as COLUMNS. Mirrors sigma_points: on Cholesky failure, retry with
// P + 1e-9*I (the jitter is applied before scaling, exactly as in Python). If the retry also
// fails, throws std::runtime_error — matches Python raising LinAlgError rather than returning
// finite nonsense.
inline Eigen::MatrixXd sigmaPoints(const Eigen::VectorXd& x,
                                   const Eigen::MatrixXd& p,
                                   double lambda) {
  const int n = static_cast<int>(x.size());
  const double scale = static_cast<double>(n) + lambda;

  Eigen::LLT<Eigen::MatrixXd> llt(scale * p);
  if (llt.info() != Eigen::Success) {
    llt.compute(scale * (p + 1e-9 * Eigen::MatrixXd::Identity(n, n)));
    if (llt.info() != Eigen::Success) {
      throw std::runtime_error(
          "sigmaPoints: covariance not positive definite even after 1e-9 jitter");
    }
  }
  const Eigen::MatrixXd chol = llt.matrixL();  // lower, matches np.linalg.cholesky

  Eigen::MatrixXd sigmas(n, 2 * n + 1);
  sigmas.col(0) = x;
  for (int i = 0; i < n; ++i) {
    sigmas.col(i + 1) = x + chol.col(i);
    sigmas.col(n + i + 1) = x - chol.col(i);
  }
  return sigmas;
}

inline Eigen::VectorXd weightedMean(const Eigen::MatrixXd& sigmas, const Eigen::VectorXd& wm) {
  return sigmas * wm;
}

// noise is optional (nullptr = none), matching weighted_covariance's `noise=None` default.
inline Eigen::MatrixXd weightedCovariance(const Eigen::MatrixXd& sigmas,
                                          const Eigen::VectorXd& mean,
                                          const Eigen::VectorXd& wc,
                                          const Eigen::MatrixXd* noise = nullptr) {
  const Eigen::MatrixXd diff = sigmas.colwise() - mean;
  Eigen::MatrixXd p = diff * wc.asDiagonal() * diff.transpose();
  if (noise != nullptr) p += *noise;
  return p;
}

}  // namespace kf_common

#endif  // KF_COMMON_UNSCENTED_HPP
