// gtest for the scalar-first SO(3) helpers. Mirrors the property tests in
// prototypes/python/tests/test_so3.py so the C++ and Python math stay in lockstep.
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "kf_common/so3.hpp"

using kf_common::boxminus;
using kf_common::boxplus;
using kf_common::expMap;
using kf_common::logMap;
using kf_common::skew;

namespace {

// skew(v) * w must equal the cross product v x w.
TEST(So3, SkewMatchesCrossProduct) {
  const Eigen::Vector3d v(0.3, -0.7, 1.1);
  const Eigen::Vector3d w(2.0, 1.0, -0.5);
  EXPECT_TRUE(skew(v) * w == v.cross(w) || (skew(v) * w - v.cross(w)).norm() < 1e-15);
}

// skew is antisymmetric: S == -S^T.
TEST(So3, SkewIsAntisymmetric) {
  const Eigen::Matrix3d s = skew(Eigen::Vector3d(1.0, 2.0, 3.0));
  EXPECT_LT((s + s.transpose()).norm(), 1e-15);
}

// exp then log returns the original rotation vector (away from the pi wrap).
TEST(So3, ExpLogRoundTrip) {
  const Eigen::Vector3d rotvec(0.4, -0.9, 0.2);
  const Eigen::Vector3d recovered = logMap(expMap(rotvec));
  EXPECT_LT((recovered - rotvec).norm(), 1e-12);
}

// The small-angle branch agrees with the general formula for tiny rotations.
TEST(So3, ExpLogRoundTripSmallAngle) {
  const Eigen::Vector3d rotvec(1e-9, -2e-9, 0.5e-9);
  const Eigen::Vector3d recovered = logMap(expMap(rotvec));
  EXPECT_LT((recovered - rotvec).norm(), 1e-15);
}

// boxminus(boxplus(q, dtheta), q) == dtheta (manifold round-trip).
TEST(So3, BoxplusBoxminusRoundTrip) {
  const Eigen::Quaterniond q = expMap(Eigen::Vector3d(0.1, -0.2, 0.3));
  const Eigen::Vector3d dtheta(0.05, 0.15, -0.1);
  const Eigen::Vector3d recovered = boxminus(boxplus(q, dtheta), q);
  EXPECT_LT((recovered - dtheta).norm(), 1e-12);
}

// boxminus of a state with itself is zero.
TEST(So3, BoxminusZeroForIdenticalStates) {
  const Eigen::Quaterniond q = expMap(Eigen::Vector3d(0.7, -0.1, 0.4));
  EXPECT_LT(boxminus(q, q).norm(), 1e-12);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
