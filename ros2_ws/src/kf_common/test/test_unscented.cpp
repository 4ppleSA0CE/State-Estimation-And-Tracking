// gtest for the unscented transform. Mirrors
// prototypes/python/tests/test_filter_invariants.py::test_ukf_sigma_points_reconstruct_mean_and_covariance
// so the C++ and Python transforms stay in lockstep.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <stdexcept>

#include "kf_common/unscented.hpp"

using kf_common::sigmaPoints;
using kf_common::unscentedWeights;
using kf_common::weightedCovariance;
using kf_common::weightedMean;

TEST(Unscented, WeightsSumToOne) {
  const int n = 4;
  const auto w = unscentedWeights(n);
  EXPECT_EQ(w.wm.size(), 2 * n + 1);
  EXPECT_EQ(w.wc.size(), 2 * n + 1);
  // The mean weights sum to EXACTLY 1 in real arithmetic, but not in floating point: with
  // alpha=1e-3 the weights are wm[0] ~ -999999 against 2n copies of ~+1e5, so the sum is a
  // cancellation of ~1e6-magnitude terms whose ULP is 1.1641532182693481e-10 (np.spacing(1e6)).
  // numpy shows the identical 4.4e-11 residual (math.fsum confirms the exact sum is 1.0), so
  // 1e-9 is the precision ceiling here, not a slackened C++ port.
  EXPECT_NEAR(w.wm.sum(), 1.0, 1e-9);
}

// WeightsSumToOne above does NOT constrain alpha/beta/kappa: wm.sum() == (lam+n)/(n+lam) == 1
// identically for ANY lambda, and the reconstruction tests below hold for any self-consistent
// (lam, wm, wc) triple. So a wrong constant (alpha=1.0, beta=0, kappa=-1, ...) passes every other
// test in this file silently. Pin the actual Van der Merwe constants against the Python
// reference so a wrong constant is caught here instead of resurfacing as an unexplained parity
// failure downstream.
//
// Values from: python3 -c "from ukf_synthetic import unscented_weights;
// lam, wm, wc = unscented_weights(4); print(repr(lam), repr(wm[0]), repr(wm[1]), repr(wc[0]))"
TEST(Unscented, WeightsMatchPythonConstants) {
  const int n = 4;
  const auto w = unscentedWeights(n);
  EXPECT_NEAR(w.lambda, -3.999996, 1e-12);
  // wm/wc are ~1e6 in magnitude here, so 1e-6 absolute tolerance is ~1e-12 relative.
  EXPECT_NEAR(w.wm(0), -999998.9999712444, 1e-6);
  EXPECT_NEAR(w.wm(1), 124999.99999640555, 1e-6);
  EXPECT_NEAR(w.wc(0), -999995.9999722444, 1e-6);
}

TEST(Unscented, SigmaPointsReconstructMeanAndCovariance) {
  const int n = 4;
  Eigen::VectorXd x(n);
  x << 1.0, -2.0, 0.5, 3.0;
  Eigen::MatrixXd p(n, n);
  p << 2.0, 0.3, 0.0, 0.1,
       0.3, 1.5, 0.2, 0.0,
       0.0, 0.2, 1.0, 0.4,
       0.1, 0.0, 0.4, 2.5;

  const auto w = unscentedWeights(n);
  const Eigen::MatrixXd chi = sigmaPoints(x, p, w.lambda);
  ASSERT_EQ(chi.rows(), n);
  ASSERT_EQ(chi.cols(), 2 * n + 1);

  const Eigen::VectorXd mean = weightedMean(chi, w.wm);
  EXPECT_TRUE(mean.isApprox(x, 1e-9)) << mean.transpose();

  const Eigen::MatrixXd cov = weightedCovariance(chi, mean, w.wc);
  EXPECT_TRUE(cov.isApprox(p, 1e-6)) << cov;
}

TEST(Unscented, WeightedCovarianceAddsNoise) {
  const int n = 2;
  Eigen::VectorXd x(n);
  x << 0.0, 0.0;
  const Eigen::MatrixXd p = Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd noise = 3.0 * Eigen::MatrixXd::Identity(n, n);

  const auto w = unscentedWeights(n);
  const Eigen::MatrixXd chi = sigmaPoints(x, p, w.lambda);
  const Eigen::VectorXd mean = weightedMean(chi, w.wm);
  const Eigen::MatrixXd cov = weightedCovariance(chi, mean, w.wc, &noise);
  EXPECT_TRUE(cov.isApprox(p + noise, 1e-6)) << cov;
}

// A non-PSD input must not throw: sigma_points retries the Cholesky with P + 1e-9*I. This must
// assert the actual VALUE, not just shape/finiteness: a jitter-applied-AFTER-scale bug (jitter
// should be added to P before multiplying by `scale`, per Python) still produces a finite 5-column
// result here, just with sigma points ~707x too wide, and would pass a shape/finiteness-only
// check silently.
TEST(Unscented, SigmaPointsSurvivesSingularCovariance) {
  const int n = 2;
  Eigen::VectorXd x(n);
  x << 1.0, 2.0;
  Eigen::MatrixXd p = Eigen::MatrixXd::Zero(n, n);  // singular

  const auto w = unscentedWeights(n);
  const Eigen::MatrixXd chi = sigmaPoints(x, p, w.lambda);
  EXPECT_EQ(chi.cols(), 2 * n + 1);
  EXPECT_TRUE(chi.allFinite());
  // First sigma-point column offset from the mean. Verified against Python:
  //   python3 -c "import numpy as np; from ukf_synthetic import unscented_weights, sigma_points;
  //   lam, _, _ = unscented_weights(2);
  //   chi = sigma_points(np.array([1.0, 2.0]), np.zeros((2, 2)), lam);
  //   print(repr(chi[0, 1] - 1.0))"
  // gives 4.4721359504507063e-08 (NOT the "clean" sqrt((n+lam)*1e-9) = 4.472135955063879e-08:
  // the actual value has an extra ~1 ULP-near-1.0 rounding from x + chol before this subtraction).
  EXPECT_NEAR(chi(0, 1) - 1.0, 4.4721359504507063e-08, 1e-16);
}

// Both Cholesky attempts failing (even the 1e-9*I-jittered retry is not PSD) must not silently
// return finite garbage. Python raises LinAlgError in exactly this case; the C++ port must signal
// just as loudly rather than poisoning the tracker with nonsense sigma points.
TEST(Unscented, SigmaPointsThrowsWhenNotPositiveDefiniteAfterJitter) {
  const int n = 2;
  Eigen::VectorXd x(n);
  x << 0.0, 0.0;
  const Eigen::MatrixXd p = -Eigen::MatrixXd::Identity(n, n);  // negative definite

  const auto w = unscentedWeights(n);
  EXPECT_THROW(sigmaPoints(x, p, w.lambda), std::runtime_error);
}
