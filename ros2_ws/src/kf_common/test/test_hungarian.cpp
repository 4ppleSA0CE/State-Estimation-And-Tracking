// gtest for rectangular optimal assignment. The Python side uses
// scipy.optimize.linear_sum_assignment; this must produce the same TOTAL COST on every input and
// the same pairing whenever the optimum is unique (see the tie caveat in the Stage 5B spec).
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "kf_common/hungarian.hpp"

using kf_common::hungarian;

namespace {

double assignmentCost(const Eigen::MatrixXd& cost, const std::vector<int>& assign) {
  double total = 0.0;
  for (int r = 0; r < static_cast<int>(assign.size()); ++r)
    if (assign[r] >= 0) total += cost(r, assign[r]);
  return total;
}

// Brute force over all injective row->col maps. Only for tiny matrices. Correct for n > m
// (more rows than columns) by recursing on the transpose: the minimum assignment cost is
// invariant under transposition, and the n <= m case below already handles rows/cols that are
// necessarily left unmatched.
double bruteForceMinCost(const Eigen::MatrixXd& cost) {
  const int n = static_cast<int>(cost.rows());
  const int m = static_cast<int>(cost.cols());
  if (n > m) return bruteForceMinCost(cost.transpose().eval());

  std::vector<int> cols(m);
  std::iota(cols.begin(), cols.end(), 0);
  double best = std::numeric_limits<double>::infinity();
  do {
    double total = 0.0;
    for (int r = 0; r < n; ++r) total += cost(r, cols[r]);
    best = std::min(best, total);
  } while (std::next_permutation(cols.begin(), cols.end()));
  return best;
}

// Mirrors the real caller contract (see associate_from_cost in tracking/association.py): the
// solver knows nothing about BIG_COST. Callers get a full raw assignment back and are
// responsible for filtering it themselves, treating any pair whose cost does not clear the
// gate as unmatched.
std::vector<int> filterGated(const Eigen::MatrixXd& cost, const std::vector<int>& assign,
                              double bigCost) {
  std::vector<int> matched = assign;
  for (int r = 0; r < static_cast<int>(matched.size()); ++r) {
    if (matched[r] >= 0 && !(cost(r, matched[r]) < bigCost)) matched[r] = -1;
  }
  return matched;
}

}  // namespace

TEST(Hungarian, KnownThreeByThreeOptimum) {
  Eigen::MatrixXd cost(3, 3);
  cost << 4.0, 1.0, 3.0,
          2.0, 0.0, 5.0,
          3.0, 2.0, 2.0;
  // Unique optimum: (0->1, 1->0, 2->2) = 1 + 2 + 2 = 5.
  const std::vector<int> assign = hungarian(cost);
  ASSERT_EQ(assign.size(), 3u);
  EXPECT_EQ(assign[0], 1);
  EXPECT_EQ(assign[1], 0);
  EXPECT_EQ(assign[2], 2);
  EXPECT_NEAR(assignmentCost(cost, assign), 5.0, 1e-12);
}

TEST(Hungarian, WideMatrixAssignsEveryRow) {
  Eigen::MatrixXd cost(2, 4);
  cost << 9.0, 1.0, 9.0, 9.0,
          9.0, 9.0, 9.0, 2.0;
  const std::vector<int> assign = hungarian(cost);
  ASSERT_EQ(assign.size(), 2u);
  EXPECT_EQ(assign[0], 1);
  EXPECT_EQ(assign[1], 3);
}

TEST(Hungarian, TallMatrixLeavesExtraRowsUnassigned) {
  Eigen::MatrixXd cost(4, 2);
  cost << 1.0, 9.0,
          9.0, 1.0,
          8.0, 8.0,
          8.0, 8.0;
  const std::vector<int> assign = hungarian(cost);
  ASSERT_EQ(assign.size(), 4u);
  EXPECT_EQ(assign[0], 0);
  EXPECT_EQ(assign[1], 1);
  int unassigned = 0;
  for (int r = 0; r < 4; ++r)
    if (assign[r] < 0) ++unassigned;
  EXPECT_EQ(unassigned, 2);
}

TEST(Hungarian, MatchesBruteForceOnRandomMatrices) {
  // The corpus must be BIT-IDENTICAL on every platform, or a CI failure cannot be reproduced
  // locally. std::rand()/MatrixXd::Random fails that outright. std::mt19937 alone is not enough
  // either: the ENGINE is standard-specified, but std::uniform_real_distribution and
  // std::uniform_int_distribution are implementation-defined, and libc++ vs libstdc++ measurably
  // disagree from the same seed (different trial dimensions, different cost checksum, and even a
  // different post-run engine state because they consume different numbers of draws). So derive
  // everything from raw engine output, which IS specified.
  std::mt19937 rng(42);
  constexpr double kSpan = static_cast<double>(std::mt19937::max() - std::mt19937::min());
  const auto nextUnit = [&rng]() {                       // uniform in [0,1]
    return static_cast<double>(rng() - std::mt19937::min()) / kSpan;
  };
  const auto nextDim = [&rng]() { return 1 + static_cast<int>(rng() % 5u); };  // 1..5

  for (int trial = 0; trial < 200; ++trial) {
    const int n = nextDim();
    const int m = nextDim();  // independent of n: covers wide (n<=m) AND tall (n>m)
    Eigen::MatrixXd cost(n, m);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < m; ++c) cost(r, c) = 2.0 * nextUnit();
    const std::vector<int> assign = hungarian(cost);
    EXPECT_NEAR(assignmentCost(cost, assign), bruteForceMinCost(cost), 1e-9)
        << "trial " << trial << " n=" << n << " m=" << m;
  }
}

TEST(Hungarian, EmptyInputReturnsEmpty) {
  Eigen::MatrixXd cost(0, 0);
  EXPECT_TRUE(hungarian(cost).empty());
}

TEST(Hungarian, ThrowsOnInfCost) {
  Eigen::MatrixXd cost(2, 2);
  cost << 1.0, std::numeric_limits<double>::infinity(),
          2.0, 3.0;
  EXPECT_THROW(hungarian(cost), std::invalid_argument);
}

TEST(Hungarian, ThrowsOnNanCost) {
  Eigen::MatrixXd cost(2, 2);
  cost << 1.0, std::numeric_limits<double>::quiet_NaN(),
          2.0, 3.0;
  EXPECT_THROW(hungarian(cost), std::invalid_argument);
}

// The following mirror the real BIG_COST=1e6 gating contract used by associate_from_cost in
// tracking/association.py: the solver is handed the raw cost matrix (sentinel cells included)
// and returns a full assignment; the CALLER then discards any pair whose cost is not below
// BIG_COST. The solver itself never sees or special-cases the sentinel.

TEST(Hungarian, EntireRowAtBigCostIsUnmatchedAfterCallerFilter) {
  constexpr double kBigCost = 1e6;
  Eigen::MatrixXd cost(3, 3);
  cost << kBigCost, kBigCost, kBigCost,
               1.0,      9.0,      9.0,
               9.0,      9.0,      1.0;
  const std::vector<int> raw = hungarian(cost);
  const std::vector<int> matched = filterGated(cost, raw, kBigCost);
  ASSERT_EQ(matched.size(), 3u);
  EXPECT_EQ(matched[0], -1) << "gated row must not survive the cost < BIG_COST filter";
  EXPECT_EQ(matched[1], 0);
  EXPECT_EQ(matched[2], 2);
}

TEST(Hungarian, EntireColumnAtBigCostIsUnmatchedAfterCallerFilter) {
  constexpr double kBigCost = 1e6;
  Eigen::MatrixXd cost(3, 3);
  cost << kBigCost, 1.0, 9.0,
          kBigCost, 9.0, 1.0,
          kBigCost, 9.0, 9.0;
  const std::vector<int> raw = hungarian(cost);
  const std::vector<int> matched = filterGated(cost, raw, kBigCost);
  ASSERT_EQ(matched.size(), 3u);
  // Column 0 is entirely gated out: it must end up claimed by nobody after filtering.
  EXPECT_EQ(matched[0], 1);
  EXPECT_EQ(matched[1], 2);
  EXPECT_EQ(matched[2], -1);
}

TEST(Hungarian, AllBigCostMatrixMatchesNothingAfterCallerFilter) {
  constexpr double kBigCost = 1e6;
  Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(3, 3, kBigCost);
  const std::vector<int> raw = hungarian(cost);
  const std::vector<int> matched = filterGated(cost, raw, kBigCost);
  for (int r = 0; r < 3; ++r) EXPECT_EQ(matched[r], -1);
}

TEST(Hungarian, NegativeCostsMatchBruteForce) {
  Eigen::MatrixXd cost(3, 3);
  cost << -4.0, -1.0, -3.0,
          -2.0,  0.0, -5.0,
          -3.0, -2.0, -2.0;
  const std::vector<int> assign = hungarian(cost);
  ASSERT_EQ(assign.size(), 3u);
  EXPECT_NEAR(assignmentCost(cost, assign), bruteForceMinCost(cost), 1e-9);
}

TEST(Hungarian, AllEqualCostMatrixProducesValidPermutation) {
  constexpr double kValue = 5.0;
  constexpr int kN = 3;
  Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(kN, kN, kValue);
  const std::vector<int> assign = hungarian(cost);
  ASSERT_EQ(assign.size(), 3u);
  std::vector<bool> used(kN, false);
  for (int r = 0; r < kN; ++r) {
    ASSERT_GE(assign[r], 0);
    ASSERT_LT(assign[r], kN);
    EXPECT_FALSE(used[assign[r]]) << "column " << assign[r] << " used twice";
    used[assign[r]] = true;
  }
  EXPECT_NEAR(assignmentCost(cost, assign), kN * kValue, 1e-12);
}
