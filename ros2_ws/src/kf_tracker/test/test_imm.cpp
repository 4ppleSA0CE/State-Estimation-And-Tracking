// gtest for the IMM mixing layer. Mirrors prototypes/python/tests/test_imm_filter.py.
//
// The invariant cases (bank order, row-stochastic pi, normalized mu, coast, predicted measurement)
// all hold for a bank that is internally consistent but numerically wrong — exactly how the
// ref_to_ca covariance-permutation bug survived a 1e-9 parity test for months. The
// `MatchesPython...` pins are the real guard: full predict+update cycles pinned to numbers
// generated from prototypes/python/tracking/imm_filter.py, not hand-derived.
//
// READ BEFORE ADDING A CASE: a single-predict pin from initState() does NOT exercise ImmFilter's
// mixing arithmetic. initState() gives every mode the same (x, P), and on identical mode states
// mixing is exactly the identity, so mix() can be deleted outright and every such case still
// passes. The mixing-layer pins at the BOTTOM of this file are the ones that constrain it: they
// run 20 cycles, inject an asymmetric pi through ImmConfig::pi_override, and drive the c floor,
// the mu floor and the total-likelihood underflow guard into their branches on purpose.
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "kf_tracker/imm.hpp"

using kf_tracker::ImmConfig;
using kf_tracker::ImmFilter;
using kf_tracker::Matrix4d;
using kf_tracker::Vector4d;

namespace {
// R IS ALWAYS sigma_pos^2 * I HERE, AND THAT IS LOAD-BEARING FOR EVERY PIN BELOW.
// Python's CvModelFilter/CaModelFilter take an `r` argument and then throw it away, rebuilding R
// from sigma_pos (see the MEASUREMENT NOISE note at the top of models.hpp). So a scenario whose
// r disagrees with sigma_pos^2 compares two different filters, not two implementations of one.
// Measured on the ModeProbabilityFloor scenario: passing r = 1e-6*I while leaving sigma_pos = 1.0
// moves mu from (0.9752, 0.0248, ...) to (0.5000, 0.5000, ...) — a 40x likelihood-ratio swing,
// not a rounding artifact. Anything pinned to Python must keep the two in step.
//
// ImmFilter owns unique_ptrs, so it is move-only; C++17 guaranteed elision makes by-value fine.
ImmFilter makeTunedBank(std::vector<double> omegas,
                        double sigma_pos,
                        std::vector<double> mu0,
                        std::vector<double> pi_override) {
  ImmConfig cfg;
  cfg.dt = 0.1;
  cfg.sigma_pos = sigma_pos;
  cfg.q_accel = 0.05;
  cfg.omegas = std::move(omegas);
  cfg.mu0 = std::move(mu0);
  cfg.pi_override = std::move(pi_override);
  return ImmFilter(cfg, Eigen::Matrix2d::Identity() * (cfg.sigma_pos * cfg.sigma_pos));
}

ImmFilter makeBank(std::vector<double> omegas) {
  return makeTunedBank(std::move(omegas), 1.0, {}, {});
}

ImmFilter makeFilter() { return makeBank({0.25, -0.25}); }

// Seed every mixing-layer case from the same place so the pins differ only in configuration.
void seed(ImmFilter& f) {
  Vector4d x0;
  x0 << 0.0, 0.0, 1.0, 0.5;
  f.initState(x0, Matrix4d::Identity());
}

// z_k = (0.9k, 0.05k^2): a curving track, so the modes actually disagree and mixing has work to do.
void driveCurve(ImmFilter& f, int steps) {
  for (int k = 1; k <= steps; ++k) {
    f.predict();
    Eigen::Vector2d z;
    z << 0.9 * k, 0.05 * k * k;
    f.update(z);
  }
}
}  // namespace

TEST(Imm, BankHasFourModesInFixedOrder) {
  const ImmFilter f = makeFilter();
  EXPECT_EQ(f.numModes(), 4);
  EXPECT_EQ(f.modeNames()[0], "CV");
  EXPECT_EQ(f.modeNames()[1], "CA");
  EXPECT_EQ(f.modeNames()[2], "CT+0.25");
  EXPECT_EQ(f.modeNames()[3], "CT-0.25");
}

TEST(Imm, TransitionMatrixRowsSumToOne) {
  const ImmFilter f = makeFilter();
  const Eigen::MatrixXd pi = f.transitionMatrix();
  ASSERT_EQ(pi.rows(), 4);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(pi.row(i).sum(), 1.0, 1e-12);
  EXPECT_NEAR(pi(0, 0), 0.97, 1e-12);
  EXPECT_NEAR(pi(0, 1), (1.0 - 0.97) / 3.0, 1e-12);
}

TEST(Imm, ModeProbabilitiesStayNormalizedAndPositive) {
  ImmFilter f = makeFilter();
  Vector4d x;
  x << 0.0, 0.0, 1.0, 0.0;
  f.initState(x, Matrix4d::Identity());

  for (int k = 1; k <= 20; ++k) {
    f.predict();
    Eigen::Vector2d z;
    z << 0.1 * k, 0.0;
    f.update(z);
    const Eigen::VectorXd mu = f.modeProbabilities();
    EXPECT_NEAR(mu.sum(), 1.0, 1e-12);
    EXPECT_GT(mu.minCoeff(), 0.0);
  }
}

TEST(Imm, InitStateDoesNotAliasAcrossModes) {
  ImmFilter f = makeFilter();
  Vector4d x;
  x << 1.0, 2.0, 3.0, 4.0;
  Matrix4d p = Matrix4d::Identity() * 2.0;
  f.initState(x, p);

  // Every mode must hold its OWN copy: scribbling on the caller's buffers after initState
  // cannot move the combined estimate. (Python needed an explicit .copy() per mode here.)
  Vector4d x_seeded;
  Matrix4d p_seeded;
  f.state(x_seeded, p_seeded);
  EXPECT_TRUE(x_seeded.isApprox(x, 1e-12));
  Vector4d x_scribble = x;
  Matrix4d p_scribble = p;
  x.setConstant(-99.0);
  p.setConstant(-99.0);
  Vector4d x_post;
  Matrix4d p_post;
  f.state(x_post, p_post);
  EXPECT_TRUE(x_post.isApprox(x_seeded, 1e-12)) << x_post.transpose();
  EXPECT_TRUE(p_post.isApprox(p_seeded, 1e-12)) << p_post;

  // Advancing one step must not corrupt the caller's inputs either.
  x = x_scribble;
  p = p_scribble;
  f.predict();
  EXPECT_NEAR(x(0), 1.0, 1e-12);
  EXPECT_NEAR(p(0, 0), 2.0, 1e-12);

  Vector4d xs;
  Matrix4d ps;
  f.state(xs, ps);
  EXPECT_TRUE(xs.allFinite());
  EXPECT_TRUE(ps.allFinite());
}

// coast() is documented as doing NOTHING: predict() has already advanced every mode, so the
// coasted estimate is whatever predict() left behind. "Position advanced and the trace grew" is
// not that property — both are MORE true after a second predict, so a coast() that quietly ran
// another predict() would pass such a test. The assertion has to be EXACT equality across the
// call. Everything before the coast() is there to make the equality non-trivial: after a real
// predict the three quantities are all different from where they started.
TEST(Imm, CoastIsExactlyANoOpAfterPredict) {
  ImmFilter f = makeFilter();
  Vector4d x;
  x << 0.0, 0.0, 5.0, 0.0;
  f.initState(x, Matrix4d::Identity());
  f.predict();
  Eigen::Vector2d z;
  z << 0.5, 0.0;
  f.update(z);

  const Eigen::VectorXd mu_updated = f.modeProbabilities();
  Vector4d x_updated;
  Matrix4d p_updated;
  f.state(x_updated, p_updated);

  // The predict() that a missed detection performs: the estimate must move.
  f.predict();
  const Eigen::VectorXd mu_predicted = f.modeProbabilities();
  Vector4d x_predicted;
  Matrix4d p_predicted;
  f.state(x_predicted, p_predicted);
  EXPECT_GT(x_predicted(0), x_updated(0));                 // position advanced
  // Hoisted out of the macro: topLeftCorner<2, 2> has a comma the preprocessor would split on.
  const double trace_predicted = p_predicted.topLeftCorner<2, 2>().trace();
  const double trace_updated = p_updated.topLeftCorner<2, 2>().trace();
  EXPECT_GT(trace_predicted, trace_updated);               // uncertainty grew
  EXPECT_TRUE(mu_predicted.isApprox(mu_updated, 1e-12));   // predict() does not touch mu

  // ...and coast() must then move NOTHING, to the last bit.
  f.coast();
  Vector4d x_coasted;
  Matrix4d p_coasted;
  f.state(x_coasted, p_coasted);
  const Eigen::VectorXd mu_coasted = f.modeProbabilities();
  for (int i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(x_coasted(i), x_predicted(i)) << "x(" << i << ")";
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_DOUBLE_EQ(p_coasted(i, j), p_predicted(i, j)) << "P(" << i << "," << j << ")";
  ASSERT_EQ(mu_coasted.size(), mu_predicted.size());
  for (int i = 0; i < mu_coasted.size(); ++i)
    EXPECT_DOUBLE_EQ(mu_coasted(i), mu_predicted(i)) << "mu(" << i << ")";
}

TEST(Imm, PredictedMeasurementMatchesCombinedState) {
  ImmFilter f = makeFilter();
  Vector4d x;
  x << 3.0, -4.0, 1.0, 2.0;
  f.initState(x, Matrix4d::Identity());

  Eigen::Vector2d z_pred;
  Eigen::Matrix2d s;
  f.predictedMeasurement(z_pred, s);

  Vector4d xs;
  Matrix4d ps;
  f.state(xs, ps);
  EXPECT_NEAR(z_pred(0), xs(0), 1e-12);
  EXPECT_NEAR(z_pred(1), xs(1), 1e-12);
  EXPECT_NEAR(s(0, 0), ps(0, 0) + 1.0, 1e-12);   // R = sigma_pos^2 * I = I
}

// Reference values from prototypes/python/tracking/imm_filter.py, regenerated 2026-07-27 with:
//   cfg = IMMConfig(dt=0.1, sigma_pos=1.0, q_accel=0.05, omegas=(0.25, -0.25))
//   f = IMMFilter(cfg, np.eye(2) * cfg.sigma_pos**2)
//   f.init_state(np.array([0.0, 0.0, 1.0, 0.5]), np.eye(4))
//   f.predict(); f.update(np.array([0.12, 0.06])); x, p = f.state()
// makeFilter() above uses exactly this config — change it and these constants stop applying.
//
// TOLERANCE, measured 2026-07-27 against numpy 2.2.6: mu and P match to 1e-12 (in fact to <1e-14),
// but the combined state x needs 1e-9 — the observed max residual is 3.9e-11 on x(2). That is the
// two CT modes' unscented weighted mean, conditioned at ~2e6 (wm[0] ~ -1e6 against 2n copies of
// +1.25e5) and dispatched through the host BLAS; numpy disagrees with ITSELF by 5.8e-10 across
// equivalent formulations of it. It is NOT the mixing layer: the CV+CA-only pin below runs the
// identical mixing code with the UKF removed and is bit-exact. If x ever drifts past 1e-9, or if
// TwoModeBank starts failing, that is a real bug — check the CA inflate/deflate round trip, the
// c/mu floors, and the mixing loop's index order.
TEST(Imm, OneStepMatchesPythonReference) {
  ImmFilter f = makeFilter();
  Vector4d x0;
  x0 << 0.0, 0.0, 1.0, 0.5;
  f.initState(x0, Matrix4d::Identity());
  f.predict();
  Eigen::Vector2d z;
  z << 0.12, 0.06;
  f.update(z);

  static const double kPyX[4] = {0.11004731965499069, 0.055023659829775966,
                                 1.000843962188346, 0.50042198110886926};
  static const double kPyMu[4] = {0.2500039306887275, 0.24998838609061941,
                                  0.25000384161029854, 0.25000384161035455};
  static const double kPyP00 = 0.50249558958395846;
  static const double kPyP11 = 0.50249573459572283;

  Vector4d x;
  Matrix4d p;
  f.state(x, p);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(x(i), kPyX[i], 1e-9) << "state element " << i;
  const Eigen::VectorXd mu = f.modeProbabilities();
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(mu(i), kPyMu[i], 1e-12) << "mode " << i;
  EXPECT_NEAR(p(0, 0), kPyP00, 1e-12);
  EXPECT_NEAR(p(1, 1), kPyP11, 1e-12);
}

// The same cycle on a CV+CA-only bank (omegas empty), i.e. with the UKF's conditioning removed,
// so it pins at 1e-12 instead of 1e-9. Generated alongside the constants above with omegas=()
// and r = np.eye(2); measured C++ vs numpy 2.2.6: mu, P00, P11 bit-identical, x within 1 ulp.
//
// WHAT IT ACTUALLY PINS — and it is less than it looks. initState() gives every mode the SAME
// (x, P), and on identical mode states mixing is the IDENTITY: measured max|mixed_x - x0| is
// 0.000e+00 on this first predict (it first becomes non-zero, 1.007e-03, on the SECOND predict).
// So this case pins the per-mode CV and CA filter arithmetic, the bank ORDER, and the mu/state
// combine weights — the ref_to_ca failure mode, a self-consistent-but-wrong bank, does show up
// here — and nothing else. It does NOT exercise the mixing loop's index order, the c floor, or
// the mu floor/renormalization: c is (0.5, 0.5), mu.sum() is already 1, and neither floor is
// anywhere near engaging. Those four live in AsymmetricPiMultiStep..., UnreachableMode... and
// ModeProbabilityFloor... below; delete this case's mixing coverage from your mental model.
TEST(Imm, TwoModeBankMatchesPythonReferenceExactly) {
  ImmFilter f = makeBank({});
  ASSERT_EQ(f.numModes(), 2);
  Vector4d x0;
  x0 << 0.0, 0.0, 1.0, 0.5;
  f.initState(x0, Matrix4d::Identity());
  f.predict();
  Eigen::Vector2d z;
  z << 0.12, 0.06;
  f.update(z);

  static const double kPyX[4] = {0.11005006680078186, 0.05502503340039093,
                                 1.0010076787113524, 0.50050383935567622};
  static const double kPyMu[2] = {0.50001554483697697, 0.49998445516302309};

  Vector4d x;
  Matrix4d p;
  f.state(x, p);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(x(i), kPyX[i], 1e-12) << "state element " << i;
  const Eigen::VectorXd mu = f.modeProbabilities();
  for (int i = 0; i < 2; ++i) EXPECT_NEAR(mu(i), kPyMu[i], 1e-12) << "mode " << i;
  EXPECT_NEAR(p(0, 0), 0.50250334003918828, 1e-12);
  EXPECT_NEAR(p(1, 1), 0.50250334003911656, 1e-12);
}

// ---------------------------------------------------------------------------
// Mixing-layer pins.
//
// Everything above seeds all modes identically and then takes ONE predict, and on identical mode
// states mixing is exactly the identity — mix() could be deleted and every case above still
// passes. Two things have to change for the mixing arithmetic to be observable at all:
//
//   1. RUN PAST ONE PREDICT. The modes only disagree after their first update, so mixing does no
//      work until the second predict.
//   2. USE AN ASYMMETRIC pi AND A NON-UNIFORM mu0. ImmConfig::piMatrix() builds a symmetric
//      matrix from pi_diag, and against a symmetric pi a pi/pi^T swap — in the mixing weights or
//      in c = pi^T mu — is mathematically invisible. ImmConfig::pi_override exists for this.
//
// References regenerated 2026-07-27 from prototypes/python/tracking/imm_filter.py (numpy 2.2.6):
//   cfg = IMMConfig(dt=0.1, sigma_pos=1.0, q_accel=0.05, omegas=(), mu0=(0.8, 0.2))
//   f = IMMFilter(cfg, np.eye(2))
//   f.pi = np.array([[0.85, 0.15], [0.40, 0.60]])   # Python's equivalent of pi_override
//   f.init_state(np.array([0.0, 0.0, 1.0, 0.5]), np.eye(4))
//   for k in range(1, 21): f.predict(); f.update(np.array([0.9*k, 0.05*k*k]))
// Measured C++ vs numpy over the 20 cycles: max|dx| 3.6e-15, max|dmu| 2.9e-15, max|dP| 6.7e-16,
// so 1e-12 leaves ~300x of margin. Transposing pi alone moves x by 1.13e-2 and mu by 2.31e-1.
// ---------------------------------------------------------------------------
TEST(Imm, AsymmetricPiMultiStepMatchesPythonReference) {
  ImmFilter f = makeTunedBank({}, 1.0, {0.8, 0.2}, {0.85, 0.15, 0.40, 0.60});
  ASSERT_EQ(f.numModes(), 2);
  // Guard the guard: if pi_override ever stops reaching the mixer, the pin below is meaningless.
  EXPECT_NEAR(f.transitionMatrix()(0, 1), 0.15, 1e-15);
  EXPECT_NEAR(f.transitionMatrix()(1, 0), 0.40, 1e-15);
  EXPECT_NEAR(f.modeProbabilities()(0), 0.8, 1e-15);

  seed(f);
  driveCurve(f, 20);

  static const double kPyX[4] = {17.266905074924633, 16.153591559788349, 8.3994153779486727,
                                 9.7074192293263177};
  static const double kPyMu[2] = {0.71239787433363999, 0.28760212566636006};
  // Row-major, as printed by numpy.
  static const double kPyP[16] = {
      0.17651646152843417,  0.00013688657415516253, 0.14880879637747779, 0.0003419112366167689,
      0.0001368865741551625, 0.17670394961483066,   0.00033261802591674436, 0.1494500451343892,
      0.14880879637747782,  0.00033261802591674425, 0.22827073086150715, 0.00094498382094553306,
      0.0003419112366167689, 0.14945004513438925,   0.00094498382094553306, 0.23061440917791159};

  Vector4d x;
  Matrix4d p;
  f.state(x, p);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(x(i), kPyX[i], 1e-12) << "x(" << i << ")";
  const Eigen::VectorXd mu = f.modeProbabilities();
  for (int i = 0; i < 2; ++i) EXPECT_NEAR(mu(i), kPyMu[i], 1e-12) << "mu(" << i << ")";
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_NEAR(p(i, j), kPyP[i * 4 + j], 1e-12) << "P(" << i << "," << j << ")";
}

// The c floor. pi's second COLUMN is all zeros, i.e. no mode can transition into mode 1, so
// c(1) = sum_i pi(i,1) mu(i) is exactly 0.0 and every mixing weight for mode 1 is 0/c(1).
// The 1e-12 floor is the only thing standing between that and 0/0 = NaN, and the floored result
// is well defined: mode 1 is re-seeded with the zero state and zero covariance.
// Reference from the same driver with f.pi = np.array([[1.0, 0.0], [1.0, 0.0]]) and the default
// uniform mu0; measured C++ vs numpy: x and mu bit-identical, max|dP| 2.8e-17.
TEST(Imm, UnreachableModeFloorsTheMixingDenominator) {
  ImmFilter f = makeTunedBank({}, 1.0, {}, {1.0, 0.0, 1.0, 0.0});
  seed(f);
  driveCurve(f, 3);

  Vector4d x;
  Matrix4d p;
  f.state(x, p);
  EXPECT_TRUE(x.allFinite()) << x.transpose();
  EXPECT_TRUE(p.allFinite()) << p;
  const Eigen::VectorXd mu = f.modeProbabilities();
  EXPECT_TRUE(mu.allFinite()) << mu.transpose();

  static const double kPyX[4] = {1.55717435975198, 0.25714724585110232, 1.3812290632741198,
                                 0.54765851376579988};
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(x(i), kPyX[i], 1e-12) << "x(" << i << ")";
  EXPECT_NEAR(mu(0), 0.99999999999899991, 1e-12);
  EXPECT_NEAR(mu(1), 9.9999999999908132e-13, 1e-20);
  EXPECT_NEAR(p(0, 0), 0.27144252057474472, 1e-12);
  EXPECT_NEAR(p(1, 1), 0.27144252057153295, 1e-12);
  EXPECT_NEAR(p(2, 2), 0.95381062442644138, 1e-12);
  EXPECT_NEAR(p(3, 3), 0.95381062442316256, 1e-12);
}

// The total-likelihood underflow guard. z is 1e6 away from every mode's prediction, so every
// per-mode gaussianLikelihood() underflows to exactly 0.0 and w.sum() is 0.0 — without the
// `denom < 1e-300` branch mu becomes 0/0 = NaN and the track is dead from then on. The
// documented fallback is mu0, and mu0 is deliberately (0.8, 0.2) here so "fell back to the
// prior" cannot be confused with "kept the mu it already had" (0.7339, 0.2661 at that point).
TEST(Imm, TotalLikelihoodUnderflowFallsBackToTheModePrior) {
  ImmFilter f = makeTunedBank({}, 1.0, {0.8, 0.2}, {0.85, 0.15, 0.40, 0.60});
  seed(f);
  driveCurve(f, 3);

  const Eigen::VectorXd mu_before = f.modeProbabilities();
  EXPECT_NEAR(mu_before(0), 0.73385011649524645, 1e-12);
  EXPECT_NEAR(mu_before(1), 0.2661498835047536, 1e-12);

  f.predict();
  Eigen::Vector2d z_wild;
  z_wild << 1.0e6, -1.0e6;
  f.update(z_wild);

  const Eigen::VectorXd mu_after = f.modeProbabilities();
  ASSERT_TRUE(mu_after.allFinite()) << mu_after.transpose();
  EXPECT_DOUBLE_EQ(mu_after(0), 0.8);
  EXPECT_DOUBLE_EQ(mu_after(1), 0.2);
}

// The mu floor AND the renormalization that has to follow it. Both CT modes turn at +/-30 rad/s,
// which over dt=0.1 swings the prediction ~3 rad away from a straight-line measurement; with
// sigma_pos = 1e-3 (so R = 1e-6*I) and an equally tight P0 that is ~100 sigma, and their
// likelihoods underflow to exactly 0.0. Their raw mu is therefore 0.0 and the 1e-12 floor is
// what keeps them alive.
//
// The renormalization is the subtle half. Flooring two modes adds 2.0e-12 to a vector that
// already summed to 1, so WITHOUT the trailing `mu / mu.sum()` the mode probabilities sum to
// 1 + 2.0e-12. That is invisible to the 1e-12 sum check in
// ModeProbabilitiesStayNormalizedAndPositive and moves mu(0) by only 2e-12 * 0.975 = 2.0e-12,
// i.e. under the 1e-12 pin tolerance used elsewhere. The 1e-14 sum assertion below is what
// actually catches it; a correctly renormalized mu sums to 1 within a few ULP (measured: 0.0
// in C++, -2.2e-16 in numpy).
// References from IMMConfig(dt=0.1, sigma_pos=1e-3, q_accel=0.05, omegas=(30.0, -30.0)),
// r = np.eye(2)*1e-6, init_state([0,0,1,0.5], np.eye(4)*1e-6), one predict + update at
// z=(0.1, 0.05). Measured C++ vs numpy: max|dmu| 2.2e-16, max|dx| 2.8e-17.
TEST(Imm, ModeProbabilityFloorIsAppliedAndRenormalized) {
  ImmFilter f = makeTunedBank({30.0, -30.0}, 1e-3, {}, {});
  ASSERT_EQ(f.numModes(), 4);
  Vector4d x0;
  x0 << 0.0, 0.0, 1.0, 0.5;
  f.initState(x0, Matrix4d::Identity() * 1e-6);
  f.predict();
  Eigen::Vector2d z;
  z << 0.1, 0.05;
  f.update(z);

  const Eigen::VectorXd mu = f.modeProbabilities();
  ASSERT_TRUE(mu.allFinite()) << mu.transpose();
  // Floor engaged: both CT modes sit AT it, not at their raw 0.0.
  EXPECT_NEAR(mu(2), 9.9999999999799988e-13, 1e-20) << "CT+30 was not floored";
  EXPECT_NEAR(mu(3), 9.9999999999799988e-13, 1e-20) << "CT-30 was not floored";
  // Renormalized afterwards: the 2.0e-12 the floor injected is gone again.
  EXPECT_NEAR(mu.sum(), 1.0, 1e-14) << "mu was floored but not renormalized";
  EXPECT_NEAR(mu(0), 0.97521289537517841, 1e-12);
  EXPECT_NEAR(mu(1), 0.024787104622821448, 1e-12);

  Vector4d x;
  Matrix4d p;
  f.state(x, p);
  static const double kPyX[4] = {0.099999999999941427, 0.049999999999970714,
                                 0.99999999999748568, 0.49999999999874289};
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(x(i), kPyX[i], 1e-12) << "x(" << i << ")";
  EXPECT_NEAR(p(0, 0), 7.0066168528875295e-07, 1e-18);
  EXPECT_NEAR(p(1, 1), 7.0066168462576554e-07, 1e-18);
}
