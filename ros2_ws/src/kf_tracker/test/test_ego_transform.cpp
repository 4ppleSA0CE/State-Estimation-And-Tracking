#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <Eigen/Dense>

#include "kf_tracker/ego_transform.hpp"

namespace {

using kf_tracker::Box3D;
using kf_tracker::EgoPose;
using kf_tracker::EgoPoseBuffer;

constexpr double kPi = 3.14159265358979323846;
constexpr double kTol = 1e-12;

EgoPose egoYaw(double x, double y, double z, double psi) {
  EgoPose e;
  e.p = Eigen::Vector3d(x, y, z);
  e.q = Eigen::Quaterniond(std::cos(psi * 0.5), 0.0, 0.0, std::sin(psi * 0.5));
  e.q.normalize();
  return e;
}

Box3D box(double x, double y, double z, double yaw) {
  Box3D b;
  b.x = x; b.y = y; b.z = z; b.yaw = yaw;
  b.l = 3.9; b.w = 1.6; b.h = 1.5;
  return b;
}

void expectBev(const Box3D& got, double x, double y, double z, double yaw, const char* label) {
  EXPECT_NEAR(got.x,   x,   kTol) << label;
  EXPECT_NEAR(got.y,   y,   kTol) << label;
  EXPECT_NEAR(got.z,   z,   kTol) << label;
  EXPECT_NEAR(got.yaw, yaw, kTol) << label;
}

// ---------------------------------------------------------------------------
// Table B from the plan -- hand-computed, NOT derived from targets.py.
// B1 and B2 reduce to table A's first two rows, which is the cross-language cross-check.
// ---------------------------------------------------------------------------
TEST(EgoTransform, MatchesPinnedTableB) {
  expectBev(kf_tracker::transformToMapBev(box(10.0, 2.0, -1.7, 0.5), egoYaw(0, 0, 0, 0.0)),
            10.0, 1.7, 2.0, -0.5, "B1");
  expectBev(kf_tracker::transformToMapBev(box(1.0, 0.0, 0.0, 0.0), egoYaw(100, 200, 5, 0.0)),
            101.0, -5.0, 200.0, 0.0, "B2");
  expectBev(kf_tracker::transformToMapBev(box(10.0, 0.0, 0.0, 0.0), egoYaw(0, 0, 0, kPi / 2)),
            0.0, 0.0, 10.0, -kPi / 2, "B3");
  expectBev(kf_tracker::transformToMapBev(box(0.0, 4.0, 1.0, kPi / 4), egoYaw(5, -5, 2, -kPi / 2)),
            9.0, -3.0, -5.0, kPi / 4, "B4");
  expectBev(kf_tracker::transformToMapBev(box(2.0, 3.0, 0.5, -1.0), egoYaw(0, 0, 0, kPi)),
            -2.0, -0.5, -3.0, 1.0 - kPi, "B5");
}

TEST(EgoTransform, CarriesDimensionsAndIdentityThrough) {
  Box3D b = box(3.0, 1.0, 0.0, 0.2);
  b.score = 0.75;
  b.track_id = 17;
  const Box3D out = kf_tracker::transformToMapBev(b, egoYaw(1, 2, 3, 0.4));
  EXPECT_DOUBLE_EQ(out.l, b.l);
  EXPECT_DOUBLE_EQ(out.w, b.w);
  EXPECT_DOUBLE_EQ(out.h, b.h);
  EXPECT_DOUBLE_EQ(out.score, b.score);
  EXPECT_EQ(out.track_id, b.track_id);
}

// Sentinel: a y/z swap or a dropped negation would pass a symmetric fixture. This cannot.
TEST(EgoTransform, IsNotIdentityOnYAndZ) {
  const Box3D out = kf_tracker::transformToMapBev(box(1.0, 2.0, 3.0, 0.4), egoYaw(0, 0, 0, 0.0));
  EXPECT_NEAR(out.y, -3.0, kTol);
  EXPECT_NEAR(out.z, 2.0, kTol);
  EXPECT_NEAR(out.yaw, -0.4, kTol);
}

// Roll/pitch must reach the POSITION (full R), even though yaw composition is heading-only.
TEST(EgoTransform, PitchRotatesThePosition) {
  EgoPose e;
  e.p.setZero();
  const double pitch = 0.3;                       // about the ENU y axis
  e.q = Eigen::Quaterniond(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
  const Box3D out = kf_tracker::transformToMapBev(box(10.0, 0.0, 0.0, 0.0), e);
  EXPECT_NEAR(out.x, 10.0 * std::cos(pitch), 1e-9);      // East
  EXPECT_NEAR(out.y, 10.0 * std::sin(pitch), 1e-9);      // Down == -Up, and Up = -10 sin(p)
  EXPECT_NEAR(out.z, 0.0, 1e-9);                          // North
}

TEST(EgoTransform, HeadingComesFromTheRotationNotTheQuaternionOrder) {
  // Composed roll+yaw: the extracted heading must still be the yaw, not a mixed angle.
  const double psi = 0.9, roll = 0.2;
  EgoPose e;
  e.p.setZero();
  e.q = Eigen::Quaterniond(Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()) *
                           Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
  EXPECT_NEAR(kf_tracker::egoHeading(e), psi, 1e-9);
}

// ---------------------------------------------------------------------------
// EgoPoseBuffer
// ---------------------------------------------------------------------------
TEST(EgoPoseBuffer, FindsAnExactStampAndNothingElse) {
  EgoPoseBuffer buf(8);
  buf.insert(1000, egoYaw(1, 2, 3, 0.1));
  ASSERT_NE(buf.find(1000), nullptr);
  EXPECT_DOUBLE_EQ(buf.find(1000)->p.x(), 1.0);
  EXPECT_EQ(buf.find(999), nullptr);      // NEAREST-match would return the entry here
  EXPECT_EQ(buf.find(1001), nullptr);
}

TEST(EgoPoseBuffer, EvictsTheOldestOnOverflow) {
  EgoPoseBuffer buf(3);
  for (std::int64_t s = 1; s <= 5; ++s) buf.insert(s * 100, egoYaw(s, 0, 0, 0.0));
  EXPECT_EQ(buf.size(), 3u);
  EXPECT_EQ(buf.find(100), nullptr);
  EXPECT_EQ(buf.find(200), nullptr);
  ASSERT_NE(buf.find(300), nullptr);
  ASSERT_NE(buf.find(500), nullptr);
  EXPECT_EQ(buf.oldestStamp(), 300);
}

TEST(EgoPoseBuffer, OutOfOrderInsertStillEvictsTheOldestByStamp) {
  EgoPoseBuffer buf(2);
  buf.insert(500, egoYaw(5, 0, 0, 0.0));
  buf.insert(100, egoYaw(1, 0, 0, 0.0));
  buf.insert(300, egoYaw(3, 0, 0, 0.0));   // evicts stamp 100, the oldest -- not the first-in
  EXPECT_EQ(buf.find(100), nullptr);
  EXPECT_NE(buf.find(300), nullptr);
  EXPECT_NE(buf.find(500), nullptr);
}

// Regression, found by review 2026-07-28: an insert() that evicts BEFORE inserting survives
// every other test here, yet discards a NEWER pose to admit an older one. oldestStamp() then
// regresses (reporting coverage back to the stale stamp while a hole exists above it), and
// detection_transform_node uses exactly that value to decide a frame is permanently
// unmatchable -- so the bug would misclassify an unresolvable frame as still-waiting.
// OutOfOrderInsertStillEvictsTheOldestByStamp looks like it covers this and does not: its
// out-of-order stamp is not below the minimum, so both orderings coincide there.
TEST(EgoPoseBuffer, InsertOlderThanEverythingHeldDoesNotEvictANewerPose) {
  EgoPoseBuffer buf(2);
  buf.insert(500, egoYaw(5, 0, 0, 0.0));
  buf.insert(300, egoYaw(3, 0, 0, 0.0));
  buf.insert(100, egoYaw(1, 0, 0, 0.0));   // older than both -- must not displace 300
  EXPECT_EQ(buf.find(100), nullptr);
  ASSERT_NE(buf.find(300), nullptr);
  ASSERT_NE(buf.find(500), nullptr);
  EXPECT_EQ(buf.oldestStamp(), 300);
}

TEST(EgoPoseBuffer, OverwriteOfTheSameStampDoesNotGrow) {
  EgoPoseBuffer buf(4);
  buf.insert(700, egoYaw(1, 0, 0, 0.0));
  buf.insert(700, egoYaw(2, 0, 0, 0.0));
  EXPECT_EQ(buf.size(), 1u);
  EXPECT_DOUBLE_EQ(buf.find(700)->p.x(), 2.0);
}

TEST(EgoPoseBuffer, EmptyBufferIsSafe) {
  const EgoPoseBuffer buf(4);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(buf.find(1), nullptr);
  EXPECT_EQ(buf.oldestStamp(), std::numeric_limits<std::int64_t>::min());
  EXPECT_FALSE(buf.full());
}

TEST(EgoPoseBuffer, ZeroCapacityIsClampedToOne) {
  EgoPoseBuffer buf(0);
  buf.insert(5, egoYaw(1, 0, 0, 0.0));
  EXPECT_EQ(buf.size(), 1u);
  EXPECT_TRUE(buf.full());
}

}  // namespace
