// Rectangular linear sum assignment (Jonker-Volgenant, O(n^3) shortest augmenting path with
// potentials). The C++ counterpart of scipy.optimize.linear_sum_assignment, which the Python
// tracker uses in tracking/association.py.
//
// Returns assignment[row] = col, or -1 when a row is left unassigned (only possible when there
// are more rows than columns). Any optimal assignment has the same total cost; when the optimum
// is NOT unique the chosen pairing may differ from scipy's. The Stage 5B reference scenario is
// built tie-free so the parity gate can compare ids directly.
#ifndef KF_COMMON_HUNGARIAN_HPP
#define KF_COMMON_HUNGARIAN_HPP

#include <Eigen/Dense>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace kf_common {
namespace detail {

// Core solver, requires rows <= cols. Internally 1-indexed (index 0 is the sentinel used by the
// augmenting-path walk), which is why the arrays are sized +1.
inline std::vector<int> hungarianSquareOrWide(const Eigen::MatrixXd& a) {
  const int n = static_cast<int>(a.rows());
  const int m = static_cast<int>(a.cols());
  constexpr double kInf = std::numeric_limits<double>::infinity();

  std::vector<double> u(n + 1, 0.0);   // row potentials
  std::vector<double> v(m + 1, 0.0);   // column potentials
  std::vector<int> p(m + 1, 0);        // p[col] = row matched to col (1-indexed; 0 = free)
  std::vector<int> way(m + 1, 0);      // predecessor column on the augmenting path

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, kInf);
    std::vector<char> used(m + 1, 0);

    do {
      used[j0] = 1;
      const int i0 = p[j0];
      double delta = kInf;
      int j1 = 0;
      for (int j = 1; j <= m; ++j) {
        if (used[j]) continue;
        const double cur = a(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    // Walk the augmenting path back, flipping matches.
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> assign(n, -1);
  for (int j = 1; j <= m; ++j)
    if (p[j] != 0) assign[p[j] - 1] = j - 1;
  return assign;
}

}  // namespace detail

// Minimum-cost assignment of rows to columns. assignment[row] = col, or -1 if unassigned.
// Throws std::invalid_argument if cost contains inf or NaN: the shortest-augmenting-path
// search below never terminates on non-finite reduced costs, so this is validated once here
// rather than inside the solver.
inline std::vector<int> hungarian(const Eigen::MatrixXd& cost) {
  if (!cost.allFinite()) {
    throw std::invalid_argument("hungarian: cost matrix contains inf or NaN");
  }

  const int n = static_cast<int>(cost.rows());
  const int m = static_cast<int>(cost.cols());
  if (n == 0 || m == 0) return std::vector<int>(static_cast<std::size_t>(n), -1);

  if (n <= m) return detail::hungarianSquareOrWide(cost);

  // More rows than columns: solve the transpose, then invert the mapping.
  const std::vector<int> t = detail::hungarianSquareOrWide(cost.transpose().eval());
  std::vector<int> assign(n, -1);
  for (int col = 0; col < static_cast<int>(t.size()); ++col)
    if (t[col] >= 0) assign[t[col]] = col;
  return assign;
}

}  // namespace kf_common

#endif  // KF_COMMON_HUNGARIAN_HPP
