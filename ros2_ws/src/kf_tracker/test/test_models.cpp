// gtest for the IMM per-mode filters. The CA round-trip case is the important one: the equivalent
// Python assertion (test_filter_invariants.py::test_ca_reference_conversion_is_an_exact_round_trip)
// is what exposed a covariance-permutation bug in imm_synthetic.ref_to_ca, and the C++
// inflate/deflate pair is the same trap.
//
// The `PinnedTo*Python` cases in the middle are the real parity guard: structural invariants
// (round trip, symmetry, positive likelihood) all hold for a filter that is internally consistent
// but numerically wrong, which is exactly how the ref_to_ca bug survived for months. Those
// expectations were generated from prototypes/python/imm_synthetic.py, not hand-derived.
//
// The cases after those pin the things a value comparison structurally CANNOT see: the omega
// guard's threshold, the likelihood floor's branch, the Joseph form (algebraically identical to
// P = (I-KH)P, so only a saturated gain separates them) and the .eval() in the CT symmetrize
// (only observable on a non-symmetric P). See CANNOT BE TESTED at the bottom of the file for the
// two remaining CT variations that provably have no such handle.
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "kf_tracker/models.hpp"

using kf_tracker::CaModelFilter;
using kf_tracker::CtModelFilter;
using kf_tracker::CvModelFilter;
using kf_tracker::gaussianLikelihood;
using kf_tracker::Matrix4d;
using kf_tracker::Vector4d;

TEST(Models, CaReferenceRoundTripIsExact) {
  CaModelFilter f(0.1, 2.0, Eigen::Matrix2d::Identity() * 0.25);
  Vector4d x;
  x << 10.0, 20.0, 1.0, 2.0;
  Matrix4d p = Matrix4d::Zero();
  p.diagonal() << 1.0, 2.0, 100.0, 200.0;   // distinct per axis so any permutation shows

  f.setState(x, p);
  Vector4d x_back;
  Matrix4d p_back;
  f.refState(x_back, p_back);

  EXPECT_TRUE(x_back.isApprox(x, 1e-12)) << x_back.transpose();
  EXPECT_TRUE(p_back.isApprox(p, 1e-12)) << p_back;
}

TEST(Models, CvPredictAdvancesPositionByVelocityTimesDt) {
  const double dt = 0.1;
  CvModelFilter f(dt, 2.0, Eigen::Matrix2d::Identity() * 0.25);
  Vector4d x;
  x << 0.0, 0.0, 3.0, -1.0;
  f.setState(x, Matrix4d::Identity());
  f.predict();

  Vector4d x_out;
  Matrix4d p_out;
  f.refState(x_out, p_out);
  EXPECT_NEAR(x_out(0), 0.3, 1e-12);
  EXPECT_NEAR(x_out(1), -0.1, 1e-12);
  EXPECT_NEAR(x_out(2), 3.0, 1e-12);
  EXPECT_NEAR(x_out(3), -1.0, 1e-12);
}

TEST(Models, CtWithNearZeroOmegaMatchesStraightLine) {
  const double dt = 0.1;
  CtModelFilter f(dt, 1e-12, 2.0, Eigen::Matrix2d::Identity() * 0.25);
  Vector4d x;
  x << 0.0, 0.0, 3.0, -1.0;
  f.setState(x, Matrix4d::Identity() * 0.01);
  f.predict();

  Vector4d x_out;
  Matrix4d p_out;
  f.refState(x_out, p_out);
  EXPECT_NEAR(x_out(0), 0.3, 1e-6);
  EXPECT_NEAR(x_out(1), -0.1, 1e-6);
}

TEST(Models, UpdateReturnsPositiveLikelihoodAndPullsTowardMeasurement) {
  CvModelFilter f(0.1, 2.0, Eigen::Matrix2d::Identity() * 0.25);
  Vector4d x;
  x << 0.0, 0.0, 0.0, 0.0;
  f.setState(x, Matrix4d::Identity() * 4.0);

  Eigen::Vector2d z;
  z << 1.0, 1.0;
  const double like = f.update(z);
  EXPECT_GT(like, 0.0);

  Vector4d x_out;
  Matrix4d p_out;
  f.refState(x_out, p_out);
  EXPECT_GT(x_out(0), 0.0);
  EXPECT_LT(x_out(0), 1.0);
}

TEST(Models, GaussianLikelihoodMatchesClosedForm) {
  Eigen::Vector2d y;
  y << 0.0, 0.0;
  const Eigen::Matrix2d s = Eigen::Matrix2d::Identity();
  // N(0; 0, I) in 2D = 1 / (2*pi)
  EXPECT_NEAR(gaussianLikelihood(y, s), 1.0 / (2.0 * M_PI), 1e-12);
}

TEST(Models, CovariancesStaySymmetricAfterPredictUpdate) {
  const Eigen::Matrix2d r = Eigen::Matrix2d::Identity() * 0.25;
  std::vector<std::unique_ptr<kf_tracker::ModelFilter>> bank;
  bank.emplace_back(std::make_unique<CvModelFilter>(0.1, 2.0, r));
  bank.emplace_back(std::make_unique<CaModelFilter>(0.1, 2.0, r));
  bank.emplace_back(std::make_unique<CtModelFilter>(0.1, 0.25, 2.0, r));

  Vector4d x;
  x << 1.0, 2.0, 3.0, 4.0;
  Eigen::Vector2d z;
  z << 1.2, 2.1;

  for (auto& f : bank) {
    f->setState(x, Matrix4d::Identity());
    f->predict();
    f->update(z);
    Vector4d xo;
    Matrix4d po;
    f->refState(xo, po);
    EXPECT_TRUE(po.isApprox(po.transpose(), 1e-9)) << po;
    EXPECT_GT(po.determinant(), 0.0);
  }
}

// ---------------------------------------------------------------------------
// Parity pins against real numbers from prototypes/python/imm_synthetic.py.
//
// Regenerate with:
//   cd prototypes/python && python3 -c "
//   import numpy as np
//   from imm_synthetic import CvModelFilter, CaModelFilter, CtModelFilter
//   r = np.eye(2) * 0.25
//   for name, f in (('CV', CvModelFilter(0.1, 0.5, 2.0, r)),
//                   ('CA', CaModelFilter(0.1, 0.5, 2.0, r)),
//                   ('CT', CtModelFilter(0.1, 0.25, 2.0, r))):
//       f.set_state(np.array([1.,2.,3.,4.]), np.diag([1.,2.,5.,7.]))
//       f.predict()
//       like = f.update(np.array([1.35, 2.45]))
//       x, p = f.ref_state()
//       print(name, ['%.17g' % v for v in np.asarray(x).ravel()], '%.17g' % like)
//       print(['%.17g' % v for v in np.asarray(p).ravel()])"
// ---------------------------------------------------------------------------
namespace {

constexpr double kPinTol = 1e-12;

// CT's state pin is looser, and NOT because the port is sloppy. The unscented mean is
// `sigmas @ wm`, and with the van der Merwe scaling used here (alpha=1e-3, n=4) lambda is
// -3.999996, so wm(0) ~ -1.0e6 and wm(1..8) ~ +1.25e5. Terms of magnitude ~3e6 cancel down to a
// result of ~1, i.e. the sum has a condition number of ~1e6 and carries ~1e-10 of rounding noise
// no matter who evaluates it. Two independent confirmations:
//
//   * wm.sum() - 1.0 == 4.3655745685100555e-11. The weights themselves only sum to one to 4e-11,
//     so even an EXACT dot product is off by ~4e-11 * |x| here. This is a property of the
//     weights, not of any implementation.
//   * Evaluating the identical expression five legitimate ways in numpy alone (`@`/BLAS,
//     einsum, (chi*wm).sum(1), a Python loop, math.fsum) spans 4.7e-10 — wider than the
//     C++-vs-numpy gap being tolerated here.
//
// WHAT IS AND IS NOT BIT-EXACT — re-measured 2026-07-27 (g++ 11.4 aarch64 vs numpy 2.2.6),
// because an earlier version of this comment claimed more than the numbers support and the next
// person to see the gate wobble will cite whatever is written here:
//   * Sigma points ARE bit-identical to numpy. Confirmed, max|d| = 0.
//   * Propagated sigma points are bit-identical FOR THIS PIN (max|d| = 0), but that is luck, not
//     a property. ctPropagate compiles to exactly six fmadd/fmsub instructions at -O2 (verified
//     by objdump; FP contraction is on by default in both GCC and clang, and every aarch64 has
//     the instructions). Rebuilding a 24-scenario x 30-step CT sweep with -ffp-contract=off
//     moves the state by up to 7.4e-9 against the contracted build. The shipped test binaries
//     carry no -O flag at all (colcon passes only -Wall -Wextra -Wpedantic -std=gnu++17), so
//     they see the uncontracted form — do not let a future -O3 be a surprise.
//   * A Neumaier-compensated sum in C++ DOES reproduce Python's math.fsum bit-for-bit on all
//     four components (component 2: 2.8990729653160088 both sides). An earlier note recorded a
//     mismatch here; it is not reproducible with a Neumaier loop that rounds each product before
//     accumulating, and most likely came from a variant that let FMA fuse the product into the
//     accumulation, which defeats the compensation.
//   * P matches numpy to 8.9e-16 on this pin, not "~1e-15" in general — see kCtCovTol below.
// The CONCLUSION is unchanged and was re-checked the honest way: with all six fmadds removed by
// -ffp-contract=off, the CT state gap against numpy is still 1.8e-10 on this pin and 8.1e-9
// across the sweep. It is conditioning, not contraction and not a port defect. Only the
// summation ORDER differs, and it cannot be reconciled (numpy dispatches to the host BLAS).
constexpr double kCtStateTol = 1e-9;

// CT gets its own covariance tolerance; CV and CA stay bit-exact at kPinTol.
// Measured C++ (g++ 11.4, the flags colcon actually uses) vs numpy 2.2.6:
//   * this one-step pin:                                        max|dP| = 8.9e-16
//   * the same filter driven 60 steps:                          max|dP| = 1.3e-12 (peaks at step 3)
//   * 24 different (x0, P0, omega) scenarios x 30 steps each:   max|dP| = 1.1e-12
// kPinTol = 1e-12 therefore has under 2x of headroom against the trajectories this same code
// takes elsewhere, which is a flake waiting to happen. 1e-11 keeps ~8x over the worst figure
// measured anywhere and still 4 orders tighter than kCtStateTol.
constexpr double kCtCovTol = 1e-11;

// One predict + one update from a common seed, then compare the reference-space output.
void checkPin(kf_tracker::ModelFilter& f,
              const double (&x_expected)[4],
              const double (&p_expected)[16],
              double like_expected,
              double x_tol = kPinTol,
              double p_tol = kPinTol) {
  Vector4d x0;
  x0 << 1.0, 2.0, 3.0, 4.0;
  Matrix4d p0 = Matrix4d::Zero();
  p0.diagonal() << 1.0, 2.0, 5.0, 7.0;
  Eigen::Vector2d z;
  z << 1.35, 2.45;

  f.setState(x0, p0);
  f.predict();
  const double like = f.update(z);

  Vector4d x;
  Matrix4d p;
  f.refState(x, p);

  for (int i = 0; i < 4; ++i) EXPECT_NEAR(x(i), x_expected[i], x_tol) << "x(" << i << ")";
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      // Python is row-major, so index [i*4 + j].
      EXPECT_NEAR(p(i, j), p_expected[i * 4 + j], p_tol) << "P(" << i << "," << j << ")";
    }
  }
  EXPECT_NEAR(like, like_expected, kPinTol) << "likelihood";
}

}  // namespace

TEST(Models, CvPinnedToPythonReference) {
  CvModelFilter f(0.1, 2.0, Eigen::Matrix2d::Identity() * 0.25);
  const double x_expected[4] = {1.3403849851928773, 2.4446121850822182, 3.0192684896734741,
                                4.0151074330294607};
  const double p_expected[16] = {
      0.20192492596438599,  0.0,                 0.096342448367370465, 0.0,
      0.0,                  0.22306092541109027, 0.0,                  0.075537165147302882,
      0.096342448367370478, 0.0,                 4.8269297334717889,   0.0,
      0.0,                  0.075537165147302882, 0.0,                 6.8081937889269621};
  checkPin(f, x_expected, p_expected, 0.091503955273689971);
}

TEST(Models, CaPinnedToPythonReference) {
  CaModelFilter f(0.1, 2.0, Eigen::Matrix2d::Identity() * 0.25);
  const double x_expected[4] = {1.3403859095890938, 2.4446124753520748, 3.0193627780875651,
                                4.0151604943592618};
  const double p_expected[16] = {
      0.20192954794546888,  0.0,                  0.09681389043782565, 0.0,
      0.0,                  0.22306237676037366,  0.0,                 0.075802471796308474,
      0.096813890437825664, 0.0,                  4.8750168246582186,  0.0,
      0.0,                  0.075802471796308474, 0.0,                 6.8566918443651872};
  checkPin(f, x_expected, p_expected, 0.091497102831530897);
}

TEST(Models, CtPinnedToPythonReference) {
  CtModelFilter f(0.1, 0.25, 2.0, Eigen::Matrix2d::Identity() * 0.25);
  const double x_expected[4] = {1.3413415450981421, 2.4442045464435274, 2.9158592890068262,
                                4.0907180906204994};
  const double p_expected[16] = {
      0.20192494421952104,    5.1795958892281646e-06, 0.096260787084435279,
      0.0040707512923969687,  5.1795958892281646e-06, 0.22306084623950576,
      -0.0025689523235407352, 0.075488687576879876,   0.096260787084435279,
      -0.0025689523235407352, 4.8282635550164779,     -0.050960365090571141,
      0.0040707512923969687,  0.075488687576879876,   -0.050960365090571141,
      6.8068692930949979};
  checkPin(f, x_expected, p_expected, 0.09151294631785481, kCtStateTol, kCtCovTol);
}

// Guards the diagnosis above: if someone later "fixes" the weights (e.g. raises alpha) this
// starts failing and the loosened kCtStateTol can be tightened back to kPinTol.
TEST(Models, UnscentedWeightsAreIllConditionedForTheMeanAtAlpha1e3) {
  const auto w = kf_common::unscentedWeights(kf_tracker::kRefDim);
  EXPECT_LT(w.wm(0), -9.0e5);                       // huge negative centre weight
  EXPECT_GT(std::abs(w.wm.sum() - 1.0), 1e-12);     // weights do not sum to 1 to 1e-12
  EXPECT_LT(std::abs(w.wm.sum() - 1.0), 1e-9);
}

// ---------------------------------------------------------------------------
// Guards on the branches and algebraic FORMS that a value pin cannot see.
// ---------------------------------------------------------------------------

// ctPropagate divides by omega, so the |omega| < 1e-8 straight-line branch is not a nicety: at
// omega == 0.0 exactly, sin(0)/0 and (1-cos(0))/0 are both 0/0 and the propagated position is
// NaN. CtWithNearZeroOmegaMatchesStraightLine above uses omega = 1e-12, which takes the same
// branch but never divides, so it cannot see the guard being deleted.
TEST(Models, CtAtExactlyZeroOmegaTakesTheStraightLineBranch) {
  CtModelFilter f(0.1, 0.0, 2.0, Eigen::Matrix2d::Identity() * 0.25);
  Vector4d x0;
  x0 << 1.0, 2.0, 3.0, 4.0;
  Matrix4d p0 = Matrix4d::Zero();
  p0.diagonal() << 1.0, 2.0, 5.0, 7.0;
  f.setState(x0, p0);
  f.predict();

  Vector4d x;
  Matrix4d p;
  f.refState(x, p);
  ASSERT_TRUE(x.allFinite()) << "omega == 0 divided by zero: " << x.transpose();
  ASSERT_TRUE(p.allFinite()) << "omega == 0 divided by zero:\n" << p;
  // Python: CtModelFilter(0.1, 0.0, 2.0, np.eye(2)*0.25), set_state([1,2,3,4], diag(1,2,5,7)),
  // predict(). Straight line to within the unscented mean's own conditioning.
  static const double kPy[4] = {1.2999999999301508, 2.3999999999068677, 3.0000000002328306, 4.0};
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(x(i), kPy[i], kCtStateTol) << "x(" << i << ")";
}

// The other half of the guard: the threshold must stay AT 1e-8. Raising it swallows real turns,
// silently, and no covariance or symmetry invariant notices. Two pins bracket it:
//   * omega = 1e-5 — just above 1e-8, so the turn branch must run. A 1e-3 threshold would flatten
//     it to the straight line (1.3, 2.4, 3, 4), missing x(2) by 4.0e-6 and x(3) by 3.0e-6.
//   * omega = 0.05 — a turn nobody would call negligible. A 1e-1 threshold flattens this one,
//     missing x(2) by 2.0e-2.
// Both are 3 to 7 orders past kCtStateTol.
TEST(Models, CtOmegaGuardThresholdStaysAt1e8) {
  Vector4d x0;
  x0 << 1.0, 2.0, 3.0, 4.0;
  Matrix4d p0 = Matrix4d::Zero();
  p0.diagonal() << 1.0, 2.0, 5.0, 7.0;

  // Python: CtModelFilter(0.1, 1e-5, 2.0, np.eye(2)*0.25) -> set_state -> predict()
  static const double kPySmall[4] = {1.3000001999316737, 2.3999998501967639, 2.9999959995038807,
                                     4.0000029997900128};
  // Python: CtModelFilter(0.1, 0.05, 2.0, np.eye(2)*0.25) -> set_state -> predict()
  static const double kPyReal[4] = {1.3009987478144467, 2.399248335044831, 2.9799625831656158,
                                    4.0149499368853867};
  static const double kStraight[4] = {1.3, 2.4, 3.0, 4.0};

  const double omegas[2] = {1e-5, 0.05};
  const double* expected[2] = {kPySmall, kPyReal};
  for (int c = 0; c < 2; ++c) {
    CtModelFilter f(0.1, omegas[c], 2.0, Eigen::Matrix2d::Identity() * 0.25);
    f.setState(x0, p0);
    f.predict();
    Vector4d x;
    Matrix4d p;
    f.refState(x, p);
    for (int i = 0; i < 4; ++i)
      EXPECT_NEAR(x(i), expected[c][i], kCtStateTol) << "omega=" << omegas[c] << " x(" << i << ")";
    // Belt and braces: the turn is genuinely distinguishable from the flattened result, so the
    // pin above is not being satisfied by a straight line that happens to round the same way.
    double worst = 0.0;
    for (int i = 0; i < 4; ++i) worst = std::max(worst, std::abs(expected[c][i] - kStraight[i]));
    EXPECT_GT(worst, 1e3 * kCtStateTol) << "omega=" << omegas[c] << " is not a real turn";
  }
}

// gaussianLikelihood's non-positive-determinant floor. S is singular in the first case and
// indefinite in the second; both must return EXACTLY 1e-300. Dropping the floor to 0.0 poisons
// the IMM mu update (w.sum() collapses and every mode falls back to the prior), and removing the
// branch entirely feeds log(det <= 0) to std::log and returns NaN. Neither shows up in a
// scenario where S is a healthy innovation covariance, which is every other case in this file.
TEST(Models, GaussianLikelihoodFloorsNonPositiveDeterminant) {
  Eigen::Vector2d y;
  y << 0.3, -0.4;

  Eigen::Matrix2d s_singular;
  s_singular << 1.0, 1.0, 1.0, 1.0;             // det == 0
  EXPECT_DOUBLE_EQ(gaussianLikelihood(y, s_singular), 1e-300);

  Eigen::Matrix2d s_indefinite;
  s_indefinite << 1.0, 0.0, 0.0, -1.0;          // det == -1
  EXPECT_DOUBLE_EQ(gaussianLikelihood(y, s_indefinite), 1e-300);
}

// The Joseph form. Note WHY no ordinary pin can catch its replacement by P = (I-KH)P: the two
// are algebraically IDENTICAL for any P whatsoever, symmetric or not, because the derivation
// only ever uses K = P H^T S^-1 and S = H P H^T + R. So they differ solely in floating-point
// behaviour, by ~3e-14 on a well-conditioned scenario — under any tolerance worth writing.
//
// Where they part company is exactly where Joseph earns its keep: a saturated gain. With
// P = 1e12*I against R = 1e-12*I the true posterior position variance is P R / (P + R) = 1e-12,
// but K = P/(P+R) rounds to EXACTLY 1.0, so (I-KH) is exactly zero in the position block and the
// short form returns a variance of exactly 0.0 — a covariance that is no longer positive
// definite. Joseph reaches the same number through K R K^T, which is 1e-12 and manifestly
// positive. Measured: C++ 1.000000049303807e-12 vs numpy 9.9999999999999998e-13 (4.9e-20 apart),
// against 0.0 for the short form. That is a 1e-12 gap on a 1e-16 tolerance.
TEST(Models, JosephFormKeepsThePosteriorPositiveWhenTheGainSaturates) {
  Vector4d x0;
  x0 << 1.0, 2.0, 3.0, 4.0;
  const Matrix4d p0 = Matrix4d::Identity() * 1e12;
  const Eigen::Matrix2d r_tiny = Eigen::Matrix2d::Identity() * 1e-12;
  Eigen::Vector2d z;
  z << 1.35, 2.45;

  std::vector<std::unique_ptr<kf_tracker::ModelFilter>> bank;
  bank.emplace_back(std::make_unique<CvModelFilter>(0.1, 2.0, r_tiny));
  bank.emplace_back(std::make_unique<CaModelFilter>(0.1, 2.0, r_tiny));
  const char* names[2] = {"CV", "CA"};

  for (int b = 0; b < 2; ++b) {
    bank[b]->setState(x0, p0);
    bank[b]->update(z);   // no predict: Q would mask the cancellation being probed
    Vector4d x;
    Matrix4d p;
    bank[b]->refState(x, p);
    EXPECT_GT(p(0, 0), 0.0) << names[b] << ": short-form (I-KH)P collapses this to exactly 0";
    EXPECT_GT(p(1, 1), 0.0) << names[b];
    EXPECT_NEAR(p(0, 0), 1e-12, 1e-16) << names[b];
    EXPECT_NEAR(p(1, 1), 1e-12, 1e-16) << names[b];
    // The gain really did saturate — otherwise the case proves nothing.
    EXPECT_DOUBLE_EQ(x(0), 1.35) << names[b] << ": measurement did not fully win";
    EXPECT_DOUBLE_EQ(x(1), 2.45) << names[b];
  }
}

// The .eval() in CtModelFilter::update's symmetrize step. Without it, Eigen evaluates
// p_ = 0.5*(p_ + p_.transpose()) coefficient-wise into p_ itself, so writing (i,j) clobbers the
// (j,i) that the transpose still has to read; column-major traversal leaves 0.75a + 0.25b above
// the diagonal against 0.5(a+b) below, and the result is not symmetric at all.
//
// At the normal call site p_ is already symmetric to roundoff, so the aliasing costs ~1 ULP and
// nothing notices. This case feeds it a genuinely asymmetric P instead. The perturbation is
// STRICTLY UPPER (0,3) so the lower triangle is untouched, which matters: both Eigen's LLT and
// numpy's cholesky read only the lower triangle, so sigmaPoints() still sees the original PD
// matrix and the case stays numerically well behaved. k*p_zz*k^T is symmetric, so the injected
// 1000.0 survives to the symmetrize step intact and must come out as 500.0 on both sides.
TEST(Models, CtSymmetrizeIsAliasFree) {
  Matrix4d p0 = Matrix4d::Zero();
  p0.diagonal() << 1.0, 2.0, 5.0, 7.0;
  p0(0, 3) += 1000.0;
  Vector4d x0;
  x0 << 1.0, 2.0, 3.0, 4.0;
  Eigen::Vector2d z;
  z << 1.35, 2.45;

  CtModelFilter f(0.1, 0.25, 2.0, Eigen::Matrix2d::Identity() * 0.25);
  f.setState(x0, p0);
  f.update(z);

  Vector4d x;
  Matrix4d p;
  f.refState(x, p);
  ASSERT_TRUE(p.allFinite()) << p;
  // Exactly symmetric, to the bit: 0.5*(a + b) and 0.5*(b + a) are the same double.
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j)
      EXPECT_DOUBLE_EQ(p(i, j), p(j, i)) << "P(" << i << "," << j << ") vs P(" << j << "," << i << ")";
  // And symmetric about the right value: the 1000.0/0.0 pair averages to 500.0, not to the
  // 750.0/250.0 an aliased read produces.
  EXPECT_NEAR(p(0, 3), 500.0, 1e-9);
  EXPECT_NEAR(p(3, 0), 500.0, 1e-9);
}

// ---------------------------------------------------------------------------
// CANNOT BE TESTED — two variations of CtModelFilter that no assertion at any tolerance can
// separate from the real thing. Recorded here so the next mutation sweep does not spend another
// day on them. Both fail for the SAME reason, and the reason is scale-free, so "try a bigger
// state" does not help: the systematic footprint of the change is always the same order as the
// C++-vs-numpy noise it would have to be seen through.
//
// Write eps_w = wm.sum() - 1 = 4.37e-11 (pure rounding in wm(0) = lambda/(n+lambda)), and note
// that the sigma points are symmetric about x_, so for ANY symmetric weight vector w,
// sum_i w_i chi_i = x_ * sum(w) exactly. Note also that ctPropagate is an exactly LINEAR map,
// chi_pred = A chi, so no curvature term ever appears. Finally, the mean is evaluated with a
// condition number of ~1e6, so whoever computes it carries an irreducible, summation-order
// dependent error of delta ~ 1e-10 * |x|; numpy alone spans 4.7e-10 across five formulations.
//
// 1. "CT predict uses wm where wc belongs" (the two differ only in element 0, by 3.0 - alpha^2).
//    The extra term is (wc0 - wm0) * d0 d0^T with d0 = chi_pred(0) - mean = -eps_w * A x, so the
//    signal is 3 * eps_w^2 |Ax|^2. The noise from delta is 2 * |sum_i wc_i d_i| * delta
//    = 2 * (4 eps_w |Ax|) * delta. Ratio noise/signal = 8 delta / (3 eps_w |Ax|) ~ 6, and the
//    |Ax| cancels: the noise is ~6x the signal at EVERY state magnitude. Measured: at |x| ~ 4
//    the two are bit-identical (max|P_wc - P_wm| = 0); at |x| ~ 1e4 the difference is 6.2e-11
//    while the mean's own reordering noise already moves P by ~6.7e-11.
//
// 2. "CT cross-covariance centred on the RECOMPUTED sigma mean instead of the stored x_".
//    The two centres differ by exactly eps_w * x, and the difference reaches p_xz only through
//    sum_i wc_i diff_z_i, which is itself 4 eps_w |Hx| — so the effect is second order,
//    4 eps_w^2 |x|^2, while the recomputed mean's own noise contributes delta * 4 eps_w |Hx|.
//    Ratio ~2.3, again independent of scale. Measured end to end (setState + update, no predict,
//    so x_ is bit-identical on both sides): at |x| ~ 4e5 the centring changes the posterior x by
//    6.5e-6 while C++ and numpy already disagree by 8.7e-6; at |x| ~ 4e7, 42.7 against 50.0.
//    The mutant's output is never reliably outside the pin's own uncertainty.
//
// The consolation is that neither is a correctness risk on the parity path: (1) is provably a
// no-op wherever the dynamics are linear, which the coordinated turn is, and (2) is bounded by
// eps_w^2. A state-dependent turn rate would change that for (2), and if one is ever added this
// note stops applying — see the corresponding comment in CtModelFilter::update.
// ---------------------------------------------------------------------------
