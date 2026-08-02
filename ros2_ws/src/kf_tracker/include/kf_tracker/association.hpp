// GNN data association from a PRECOMPUTED cost matrix.
//
// A 1:1 C++ port of associate_from_cost / _greedy_from_cost in
// prototypes/python/tracking/association.py. Rows are tracks, columns are detections. Pairs at or
// above `big_cost` are gated out — and, exactly as in Python, the gate is applied AFTER the solver
// runs, not before: the assignment problem is solved over the full matrix (gated pairs carry
// big_cost so the optimum avoids them when it can) and only then are big-cost pairs dropped. A
// pre-filter would change which pairing the solver returns whenever every option for some row is
// gated out.
//
// kf_common::hungarian throws std::invalid_argument on a non-finite cost matrix. That is
// deliberately NOT caught here: policy for a bad frame lives at the ROS callback (see L4 in the
// C++ tracker port plan). big_cost is a large FINITE sentinel precisely so gating never trips it.
#ifndef KF_TRACKER_ASSOCIATION_HPP
#define KF_TRACKER_ASSOCIATION_HPP

#include <algorithm>   // std::sort
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "kf_common/hungarian.hpp"
#include "kf_tracker/models.hpp"   // Vector2d / Matrix2d for mahalanobisSq

namespace kf_tracker {

inline constexpr double kBigCost = 1e6;   // gated-out pairs; mirrors association.BIG_COST

struct Assignment {
  std::vector<std::pair<int, int>> matches;   // (track_row, detection_col)
  std::vector<int> unmatched_cols;            // detections with no track
  std::vector<int> unmatched_rows;            // tracks with no detection
};

namespace detail {

// AB3DMOT-style greedy fallback: take pairs in ascending cost, skipping already-used rows/cols.
//
// std::sort on the (cost, i, j) tuple breaks cost ties by (i, j), which is exactly the order the
// generation loops emit pairs in — so this reproduces Python's `sorted(..., key=lambda t: t[0])`
// (a STABLE sort over the same generation order) pair for pair, ties included.
inline std::vector<std::pair<int, int>> greedyFromCost(const Eigen::MatrixXd& cost,
                                                       double big_cost) {
  const int n = static_cast<int>(cost.rows());
  const int m = static_cast<int>(cost.cols());

  std::vector<std::tuple<double, int, int>> pairs;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      if (cost(i, j) < big_cost) pairs.emplace_back(cost(i, j), i, j);
  std::sort(pairs.begin(), pairs.end());

  std::vector<char> used_r(static_cast<std::size_t>(n), 0);
  std::vector<char> used_c(static_cast<std::size_t>(m), 0);
  std::vector<std::pair<int, int>> matches;
  for (const auto& p : pairs) {
    const int i = std::get<1>(p);
    const int j = std::get<2>(p);
    if (used_r[static_cast<std::size_t>(i)] || used_c[static_cast<std::size_t>(j)]) continue;
    matches.emplace_back(i, j);
    used_r[static_cast<std::size_t>(i)] = 1;
    used_c[static_cast<std::size_t>(j)] = 1;
  }
  return matches;
}

}  // namespace detail

// Assign rows (tracks) to columns (detections). Matches come back in ROW order from the Hungarian
// branch and in ASCENDING-COST order from the greedy branch, matching Python either way.
inline Assignment associateFromCost(const Eigen::MatrixXd& cost,
                                    double big_cost = kBigCost,
                                    bool greedy = false) {
  const int n = static_cast<int>(cost.rows());
  const int m = static_cast<int>(cost.cols());

  Assignment out;
  if (n == 0 || m == 0) {   // mirrors Python's early return; hungarian() is never called
    for (int j = 0; j < m; ++j) out.unmatched_cols.push_back(j);
    for (int i = 0; i < n; ++i) out.unmatched_rows.push_back(i);
    return out;
  }

  if (greedy) {
    out.matches = detail::greedyFromCost(cost, big_cost);
  } else {
    // hungarian() returns assign[row] = col, or -1 for a row left unassigned (possible only when
    // rows > cols). Filter AFTER solving, with `<` so a pair sitting exactly ON big_cost is gated
    // out — Python's `cost[i, j] < big_cost` has the same boundary.
    const std::vector<int> assign = kf_common::hungarian(cost);
    for (int i = 0; i < n; ++i) {
      const int j = assign[static_cast<std::size_t>(i)];
      if (j >= 0 && cost(i, j) < big_cost) out.matches.emplace_back(i, j);
    }
  }

  std::vector<char> matched_r(static_cast<std::size_t>(n), 0);
  std::vector<char> matched_c(static_cast<std::size_t>(m), 0);
  for (const auto& mt : out.matches) {
    matched_r[static_cast<std::size_t>(mt.first)] = 1;
    matched_c[static_cast<std::size_t>(mt.second)] = 1;
  }
  for (int j = 0; j < m; ++j)
    if (!matched_c[static_cast<std::size_t>(j)]) out.unmatched_cols.push_back(j);
  for (int i = 0; i < n; ++i)
    if (!matched_r[static_cast<std::size_t>(i)]) out.unmatched_rows.push_back(i);
  return out;
}

// Squared Mahalanobis distance y^T S^-1 y. Python uses np.linalg.solve; Eigen's 2x2 inverse is the
// closed-form adjugate, so the two agree to roundoff rather than bit-exactly.
inline double mahalanobisSq(const Vector2d& y, const Matrix2d& s) {
  return (y.transpose() * s.inverse() * y)(0, 0);   // 1x1 expression -> scalar
}

}  // namespace kf_tracker

#endif  // KF_TRACKER_ASSOCIATION_HPP
