// gtest for association + the tracker lifecycle. Mirrors
// prototypes/python/tests/test_association_from_cost.py and test_kitti_tracker.py.
//
// L1 from the Stage 5B plan applies with full force here: mutation testing proved that value
// assertions do not constrain structure. The tracker's step() is almost entirely ORDER and
// COMPARISON DIRECTION, and both are invisible to a state pin — moving the cost-matrix build
// ahead of the predict loop leaves every filtered number bit-identical and only changes which
// detection each track was scored against. So every test below that is not a numeric pin is
// designed against a specific defect, and the defect it kills is named in its comment.
//
// Reference numbers were generated 2026-07-27 by driving the Python tracker directly:
//   PYTHONPATH=prototypes/python:prototypes/python/tracking venv/bin/python
// with numpy 2.4.5 / scipy 1.17.1, using kitti_tracker.KittiTracker and association.py.
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "kf_tracker/association.hpp"
#include "kf_tracker/tracker.hpp"

using kf_tracker::Assignment;
using kf_tracker::associateFromCost;
using kf_tracker::Box3D;
using kf_tracker::BoxTrack;
using kf_tracker::kBigCost;
using kf_tracker::kMahaGate;
using kf_tracker::KittiTracker;
using kf_tracker::mahalanobisSq;
using kf_tracker::Matrix2d;
using kf_tracker::Matrix4d;
using kf_tracker::TrackerConfig;
using kf_tracker::TrackStatus;
using kf_tracker::Vector2d;
using kf_tracker::Vector4d;

namespace {

// KITTI Car dimensions used throughout. With yaw = 0 the box extends `l` = 3.9 m along x and
// `w` = 1.6 m along z, so a 1.6 m step in z is exactly one box length of separation in BEV.
Box3D car(double x, double z, double score = 1.0) {
  Box3D b;
  b.x = x;
  b.y = 1.6;
  b.z = z;
  b.yaw = 0.0;
  b.l = 3.9;
  b.w = 1.6;
  b.h = 1.5;
  b.score = score;
  return b;
}

// The pinned lifecycle scenario. Three targets, one occlusion gap, one blank frame:
//   A  x = 0.0,  z = 10.0 + 0.6k   every frame except k = 8
//   B  x = 8.0,  z = 20.0 - 0.4k   k <= 4 and k >= 7 (occluded at 5, 6), except k = 8
//   C  x = -9.0, z = 14.0 + 0.5k   only 3 <= k <= 5
//   k = 8        zero detections
// The three lanes are 8-9 m apart in x against a 3.9 m box, so every cross pair has IoU exactly
// 0 and is gated out; the Hungarian optimum is therefore unique and the ids are reproducible.
std::vector<Box3D> scenarioFrame(int k) {
  std::vector<Box3D> dets;
  if (k == 8) return dets;                                              // deliberately empty
  dets.push_back(car(0.0, 10.0 + 0.6 * k));                             // A
  if (k <= 4 || k >= 7) dets.push_back(car(8.0, 20.0 - 0.4 * k));       // B
  if (k >= 3 && k <= 5) dets.push_back(car(-9.0, 14.0 + 0.5 * k));      // C
  return dets;
}

struct TrackRow {
  int id;
  TrackStatus status;
  unsigned hits;
  unsigned hit_streak;
  unsigned tsu;
  unsigned age;
};

void expectRows(const KittiTracker& trk, const std::vector<TrackRow>& want, int k) {
  ASSERT_EQ(trk.numTracks(), want.size()) << "track count at frame " << k;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const BoxTrack& t = trk.tracks()[i];
    EXPECT_EQ(t.id, want[i].id) << "slot " << i << " id at frame " << k;
    EXPECT_EQ(t.status, want[i].status) << "id " << t.id << " status at frame " << k;
    EXPECT_EQ(t.hits, want[i].hits) << "id " << t.id << " hits at frame " << k;
    EXPECT_EQ(t.hit_streak, want[i].hit_streak) << "id " << t.id << " hit_streak at frame " << k;
    EXPECT_EQ(t.time_since_update, want[i].tsu) << "id " << t.id << " tsu at frame " << k;
    EXPECT_EQ(t.age, want[i].age) << "id " << t.id << " age at frame " << k;
  }
}

constexpr TrackStatus kT = TrackStatus::kTentative;
constexpr TrackStatus kC = TrackStatus::kConfirmed;

}  // namespace

// ===========================================================================================
// association.hpp
// ===========================================================================================

TEST(Association, PicksTheCheapestPairing) {
  Eigen::MatrixXd cost(2, 2);
  cost << 0.1, 0.9,
          0.9, 0.2;
  const Assignment a = associateFromCost(cost, kBigCost, false);
  ASSERT_EQ(a.matches.size(), 2u);
  EXPECT_EQ(a.matches[0].first, 0);
  EXPECT_EQ(a.matches[0].second, 0);
  EXPECT_EQ(a.matches[1].first, 1);
  EXPECT_EQ(a.matches[1].second, 1);
  EXPECT_TRUE(a.unmatched_rows.empty());
  EXPECT_TRUE(a.unmatched_cols.empty());
}

TEST(Association, GatedOutPairsBecomeUnmatched) {
  Eigen::MatrixXd cost(2, 2);
  cost << 0.1, kBigCost,
          kBigCost, kBigCost;
  const Assignment a = associateFromCost(cost, kBigCost, false);
  ASSERT_EQ(a.matches.size(), 1u);
  EXPECT_EQ(a.matches[0].first, 0);
  EXPECT_EQ(a.matches[0].second, 0);
  ASSERT_EQ(a.unmatched_rows.size(), 1u);
  EXPECT_EQ(a.unmatched_rows[0], 1);
  ASSERT_EQ(a.unmatched_cols.size(), 1u);
  EXPECT_EQ(a.unmatched_cols[0], 1);
}

// DEFECT KILLED: `unmatched_cols` and `unmatched_rows` swapped, in either the struct or the fill
// loops. Both prior tests use square matrices where the swap is invisible. These two are
// rectangular AND asymmetric, so a swap produces indices that do not even exist on the other
// axis. Python (verified): 1x3 -> ([(0,0)], cols [1,2], rows []); 3x1 -> ([(0,0)], cols [],
// rows [1,2]).
TEST(Association, RowsAndColumnsAreNotInterchangeable) {
  Eigen::MatrixXd wide(1, 3);
  wide << 0.1, kBigCost, kBigCost;
  const Assignment w = associateFromCost(wide, kBigCost, false);
  ASSERT_EQ(w.matches.size(), 1u);
  EXPECT_EQ(w.matches[0].first, 0);
  EXPECT_EQ(w.matches[0].second, 0);
  EXPECT_TRUE(w.unmatched_rows.empty());
  ASSERT_EQ(w.unmatched_cols.size(), 2u);
  EXPECT_EQ(w.unmatched_cols[0], 1);
  EXPECT_EQ(w.unmatched_cols[1], 2);

  Eigen::MatrixXd tall(3, 1);
  tall << 0.1, kBigCost, kBigCost;
  const Assignment t = associateFromCost(tall, kBigCost, false);
  ASSERT_EQ(t.matches.size(), 1u);
  EXPECT_EQ(t.matches[0].first, 0);
  EXPECT_EQ(t.matches[0].second, 0);
  EXPECT_TRUE(t.unmatched_cols.empty());
  ASSERT_EQ(t.unmatched_rows.size(), 2u);
  EXPECT_EQ(t.unmatched_rows[0], 1);
  EXPECT_EQ(t.unmatched_rows[1], 2);
}

// DEFECT KILLED: the gate written as `<=` instead of `<`. Python filters with
// `cost[i, j] < big_cost`, so a pair sitting EXACTLY on big_cost is gated out. This is the exact
// value the tracker's costMatrix() fills with, so the boundary is load-bearing, not academic.
TEST(Association, CostExactlyAtBigCostIsGatedOut) {
  Eigen::MatrixXd on(1, 1);
  on << kBigCost;
  const Assignment a = associateFromCost(on, kBigCost, false);
  EXPECT_TRUE(a.matches.empty());
  ASSERT_EQ(a.unmatched_rows.size(), 1u);
  ASSERT_EQ(a.unmatched_cols.size(), 1u);

  Eigen::MatrixXd below(1, 1);
  below << kBigCost * (1.0 - 1e-12);
  const Assignment b = associateFromCost(below, kBigCost, false);
  ASSERT_EQ(b.matches.size(), 1u);

  const Assignment g = associateFromCost(on, kBigCost, true);   // same boundary in the greedy path
  EXPECT_TRUE(g.matches.empty());
  EXPECT_EQ(g.unmatched_rows.size(), 1u);
}

TEST(Association, EmptyInputsReturnEverythingUnmatched) {
  const Eigen::MatrixXd no_rows(0, 3);
  const Assignment a = associateFromCost(no_rows, kBigCost, false);
  EXPECT_TRUE(a.matches.empty());
  EXPECT_TRUE(a.unmatched_rows.empty());
  ASSERT_EQ(a.unmatched_cols.size(), 3u);
  EXPECT_EQ(a.unmatched_cols[2], 2);

  const Eigen::MatrixXd no_cols(3, 0);
  const Assignment b = associateFromCost(no_cols, kBigCost, false);
  EXPECT_TRUE(b.matches.empty());
  EXPECT_TRUE(b.unmatched_cols.empty());
  ASSERT_EQ(b.unmatched_rows.size(), 3u);
  EXPECT_EQ(b.unmatched_rows[2], 2);
}

// DEFECT KILLED: the `greedy` flag ignored (either branch always taken). A matrix on which the
// two solvers AGREE cannot see that. This one is a deliberate greedy trap: greedy commits to the
// global minimum (0,0) = 1 and is then forced into (1,1) = 100 for a total of 101, while the
// optimum is (0,1) + (1,0) = 4. Python (verified): greedy -> [(0,0),(1,1)];
// linear_sum_assignment -> [(0,1),(1,0)].
TEST(Association, GreedyAndHungarianDisagreeOnATrapMatrix) {
  Eigen::MatrixXd cost(2, 2);
  cost << 1.0, 2.0,
          2.0, 100.0;

  const Assignment h = associateFromCost(cost, kBigCost, false);
  ASSERT_EQ(h.matches.size(), 2u);
  EXPECT_EQ(h.matches[0].first, 0);
  EXPECT_EQ(h.matches[0].second, 1);
  EXPECT_EQ(h.matches[1].first, 1);
  EXPECT_EQ(h.matches[1].second, 0);

  const Assignment g = associateFromCost(cost, kBigCost, true);
  ASSERT_EQ(g.matches.size(), 2u);
  EXPECT_EQ(g.matches[0].first, 0);
  EXPECT_EQ(g.matches[0].second, 0);
  EXPECT_EQ(g.matches[1].first, 1);
  EXPECT_EQ(g.matches[1].second, 1);
}

// Greedy emits matches in ASCENDING COST order, the Hungarian branch in ROW order. Python does the
// same, and the ROS node writes /tracks in whatever order it walks these, so pin it.
// Python (verified) on [[0.2, BIG, 0.9], [BIG, BIG, 0.1]]:
//   hungarian -> ([(0,0), (1,2)], cols [1], rows [])
//   greedy    -> ([(1,2), (0,0)], cols [1], rows [])
TEST(Association, MatchOrderDiffersBetweenSolvers) {
  Eigen::MatrixXd cost(2, 3);
  cost << 0.2, kBigCost, 0.9,
          kBigCost, kBigCost, 0.1;

  const Assignment h = associateFromCost(cost, kBigCost, false);
  ASSERT_EQ(h.matches.size(), 2u);
  EXPECT_EQ(h.matches[0].first, 0);
  EXPECT_EQ(h.matches[0].second, 0);
  EXPECT_EQ(h.matches[1].first, 1);
  EXPECT_EQ(h.matches[1].second, 2);
  ASSERT_EQ(h.unmatched_cols.size(), 1u);
  EXPECT_EQ(h.unmatched_cols[0], 1);

  const Assignment g = associateFromCost(cost, kBigCost, true);
  ASSERT_EQ(g.matches.size(), 2u);
  EXPECT_EQ(g.matches[0].first, 1);        // cheapest pair first
  EXPECT_EQ(g.matches[0].second, 2);
  EXPECT_EQ(g.matches[1].first, 0);
  EXPECT_EQ(g.matches[1].second, 0);
  ASSERT_EQ(g.unmatched_cols.size(), 1u);
  EXPECT_EQ(g.unmatched_cols[0], 1);
}

TEST(Association, AllGatedYieldsNoMatchesEitherSolver) {
  const Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(2, 2, kBigCost);
  for (bool greedy : {false, true}) {
    const Assignment a = associateFromCost(cost, kBigCost, greedy);
    EXPECT_TRUE(a.matches.empty()) << "greedy=" << greedy;
    EXPECT_EQ(a.unmatched_rows.size(), 2u) << "greedy=" << greedy;
    EXPECT_EQ(a.unmatched_cols.size(), 2u) << "greedy=" << greedy;
  }
}

// Closed form, independent of the implementation: diag(2,5) gives 9/2 + 16/5 = 7.7, and the
// correlated case inverts [[2,1],[1,3]] to [[0.6,-0.2],[-0.2,0.4]] giving 0.6 - 0.8 + 1.6 = 1.4.
// Python's mahalanobis_sq (np.linalg.solve) returns 7.7 and 1.4000000000000001.
TEST(Association, MahalanobisSqMatchesClosedForm) {
  Vector2d y;
  y << 3.0, 4.0;
  Matrix2d s;
  s << 2.0, 0.0,
       0.0, 5.0;
  EXPECT_NEAR(mahalanobisSq(y, s), 7.7, 1e-12);

  Vector2d y2;
  y2 << 1.0, 2.0;
  Matrix2d s2;
  s2 << 2.0, 1.0,
        1.0, 3.0;
  EXPECT_NEAR(mahalanobisSq(y2, s2), 1.4, 1e-12);

  Vector2d zero = Vector2d::Zero();
  EXPECT_NEAR(mahalanobisSq(zero, s2), 0.0, 1e-15);
}

// ===========================================================================================
// tracker.hpp — birth
// ===========================================================================================

// DEFECT KILLED: birth moved ahead of the predict loop, and any mis-wiring of the birth
// covariance (sigma_pos^2 on position, p0_vel on velocity — note p0_vel is a VARIANCE, not a
// sigma, and lives on TrackerConfig, never on ImmConfig).
// A track born this frame must NOT have been predicted this frame: age stays 0 and the state is
// EXACTLY the detection with zero velocity. Python (verified): x = [1.25, 7.5, 0, 0],
// diag(P) = [0.25, 0.25, 10, 10], age 0, hits 1, hit_streak 1, tsu 0, tentative.
TEST(Tracker, BirthStateIsTheDetectionAndIsNotPredicted) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  const auto out = trk.step({car(1.25, 7.5)});
  EXPECT_TRUE(out.empty());                    // tentative, so not returned
  ASSERT_EQ(trk.numTracks(), 1u);

  const BoxTrack& t = trk.tracks()[0];
  EXPECT_EQ(t.id, 0);
  EXPECT_EQ(t.status, kT);
  EXPECT_EQ(t.hits, 1u);
  EXPECT_EQ(t.hit_streak, 1u);
  EXPECT_EQ(t.time_since_update, 0u);
  EXPECT_EQ(t.age, 0u);                        // <- dies if birth happens before predict()

  Vector4d x;
  Matrix4d p;
  t.immState(x, p);
  EXPECT_NEAR(x(0), 1.25, 1e-15);
  EXPECT_NEAR(x(1), 7.5, 1e-15);
  EXPECT_NEAR(x(2), 0.0, 1e-15);
  EXPECT_NEAR(x(3), 0.0, 1e-15);
  EXPECT_NEAR(p(0, 0), 0.25, 1e-15);           // sigma_pos^2, sigma_pos = 0.5
  EXPECT_NEAR(p(1, 1), 0.25, 1e-15);
  EXPECT_NEAR(p(2, 2), 10.0, 1e-15);           // p0_vel, a variance
  EXPECT_NEAR(p(3, 3), 10.0, 1e-15);

  const Box3D b = t.box();                     // geometry is carried verbatim from the detection
  EXPECT_NEAR(b.x, 1.25, 1e-15);
  EXPECT_NEAR(b.z, 7.5, 1e-15);
  EXPECT_NEAR(b.y, 1.6, 1e-15);
  EXPECT_NEAR(b.l, 3.9, 1e-15);
  EXPECT_NEAR(b.w, 1.6, 1e-15);
  EXPECT_NEAR(b.h, 1.5, 1e-15);
  EXPECT_EQ(b.track_id, 0);
  EXPECT_EQ(t.modeProbabilities().size(), 4);  // CV + CA + 2 CT
}

TEST(Tracker, BirthRequiresMinHits) {
  TrackerConfig cfg;
  cfg.min_hits = 3;
  KittiTracker trk(cfg);

  EXPECT_TRUE(trk.step({car(0.0, 10.0)}).empty());     // hit 1 -> tentative
  EXPECT_TRUE(trk.step({car(0.0, 10.0)}).empty());     // hit 2 -> tentative
  EXPECT_EQ(trk.step({car(0.0, 10.0)}).size(), 1u);    // hit 3 -> confirmed
}

// DEFECT KILLED: promotion written `hits > min_hits` instead of `hits >= min_hits` (confirms one
// frame late), or `>` in the other direction. min_hits = 4 here so the boundary is distinct from
// the default and from max_age.
TEST(Tracker, PromotionHappensOnExactlyTheMinHitsFrame) {
  TrackerConfig cfg;
  cfg.min_hits = 4;
  KittiTracker trk(cfg);
  EXPECT_EQ(trk.step({car(0.0, 10.0)}).size(), 0u);
  EXPECT_EQ(trk.step({car(0.0, 10.0)}).size(), 0u);
  EXPECT_EQ(trk.step({car(0.0, 10.0)}).size(), 0u);
  ASSERT_EQ(trk.tracks()[0].hits, 3u);
  EXPECT_EQ(trk.step({car(0.0, 10.0)}).size(), 1u);    // hits == 4 == min_hits
  EXPECT_EQ(trk.tracks()[0].status, kC);
}

// ===========================================================================================
// tracker.hpp — death, coasting, empty frames
// ===========================================================================================

// DEFECT KILLED: death written `tsu >= max_age` instead of `> max_age` (kills one frame early),
// and markMissed() never being called (tsu would stay 0 and the track would never die).
TEST(Tracker, DiesOnlyWhenMissesExceedMaxAge) {
  TrackerConfig cfg;
  cfg.min_hits = 3;
  cfg.max_age = 2;
  KittiTracker trk(cfg);
  for (int i = 0; i < 3; ++i) trk.step({car(0.0, 10.0)});
  ASSERT_EQ(trk.numTracks(), 1u);
  ASSERT_EQ(trk.tracks()[0].status, kC);

  trk.step({});                                        // miss 1
  ASSERT_EQ(trk.numTracks(), 1u);
  EXPECT_EQ(trk.tracks()[0].time_since_update, 1u);

  trk.step({});                                        // miss 2 == max_age: SURVIVES
  ASSERT_EQ(trk.numTracks(), 1u);
  EXPECT_EQ(trk.tracks()[0].time_since_update, 2u);
  EXPECT_EQ(trk.tracks()[0].status, kC);

  trk.step({});                                        // miss 3 > max_age: dies
  EXPECT_EQ(trk.numTracks(), 0u);
}

// DEFECT KILLED: step() early-returning on an empty detection frame, and time_since_update being
// reset by markMissed() rather than by update(). An empty frame must still predict (age grows),
// coast, and age the lifecycle counters. The reference .npz has deliberately blanked frames.
TEST(Tracker, EmptyFrameStillPredictsAndAgesTracks) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  for (int i = 0; i < 3; ++i) trk.step({car(0.0, 10.0 + 0.5 * i)});
  ASSERT_EQ(trk.numTracks(), 1u);
  const unsigned age_before = trk.tracks()[0].age;
  const unsigned hits_before = trk.tracks()[0].hits;
  Vector4d x_before;
  Matrix4d p_before;
  trk.tracks()[0].immState(x_before, p_before);

  const auto out = trk.step({});
  EXPECT_EQ(out.size(), 1u);                           // still confirmed, just coasting
  ASSERT_EQ(trk.numTracks(), 1u);
  const BoxTrack& t = trk.tracks()[0];
  EXPECT_EQ(t.age, age_before + 1u);                   // predict() ran
  EXPECT_EQ(t.time_since_update, 1u);                  // markMissed() ran
  EXPECT_EQ(t.hit_streak, 0u);                         // streak broken
  EXPECT_EQ(t.hits, hits_before);                      // no update() happened

  Vector4d x_after;
  Matrix4d p_after;
  t.immState(x_after, p_after);
  EXPECT_GT(x_after(1), x_before(1));                  // coasted forward along z
  // Hoisted out of the macro: topLeftCorner<2, 2> has a comma the preprocessor would split on.
  const double pos_trace_after = p_after.topLeftCorner<2, 2>().trace();
  const double pos_trace_before = p_before.topLeftCorner<2, 2>().trace();
  EXPECT_GT(pos_trace_after, pos_trace_before);        // uncertainty grew, nothing corrected it
}

// DEFECT KILLED: time_since_update reset on miss instead of on update. A track that is seen every
// frame must hold tsu == 0 forever; if the reset moved to markMissed(), tsu would climb on every
// update and the track would die at frame 3 with max_age = 2.
TEST(Tracker, ContinuouslySeenTrackNeverAccumulatesMisses) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  for (int k = 0; k < 10; ++k) {
    trk.step({car(0.0, 10.0 + 0.5 * k)});
    ASSERT_EQ(trk.numTracks(), 1u) << "track died at frame " << k;
    EXPECT_EQ(trk.tracks()[0].time_since_update, 0u) << "frame " << k;
    EXPECT_EQ(trk.tracks()[0].hit_streak, static_cast<unsigned>(k + 1)) << "frame " << k;
  }
}

// ===========================================================================================
// tracker.hpp — cost matrix: predict ordering, gate direction, cost sign, solver flag
// ===========================================================================================

// DEFECT KILLED: the cost matrix built BEFORE the predict loop. This is the defect the plan calls
// out as silently shifting every association by one frame, and it changes no filtered value at
// all — only which detection each track is scored against — so no state pin can see it.
//
// Setup: one track warmed up to ~1 m/frame along z, then a frame carrying TWO detections, one
// sitting on the track's STALE (pre-predict) position z = 17.0 and one on its PREDICTED position
// z = 18.0. Correct code scores against the prediction and takes z = 18.0, leaving z = 17.0 to
// birth a new track; the defect takes z = 17.0 and births at z = 18.0. A newborn's box is EXACTLY
// its detection, so the discriminator is exact rather than a tolerance.
// Python (verified): id 0 -> box.z 17.845764995098378 (confirmed, hits 9, age 8),
// id 1 -> box.z exactly 17.0 (tentative, hits 1, age 0).
TEST(Tracker, CostIsScoredAgainstThePredictedBoxNotTheStaleOne) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  for (int k = 0; k < 8; ++k) trk.step({car(0.0, 10.0 + 1.0 * k)});
  ASSERT_EQ(trk.numTracks(), 1u);

  const auto out = trk.step({car(0.0, 17.0), car(0.0, 18.0)});   // [0] = stale, [1] = predicted
  ASSERT_EQ(trk.numTracks(), 2u);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0]->id, 0);

  const BoxTrack& kept = trk.tracks()[0];
  const BoxTrack& born = trk.tracks()[1];
  ASSERT_EQ(kept.id, 0);
  ASSERT_EQ(born.id, 1);
  EXPECT_EQ(kept.hits, 9u);
  EXPECT_EQ(kept.time_since_update, 0u);
  EXPECT_EQ(born.hits, 1u);
  EXPECT_EQ(born.age, 0u);

  // The whole test: the LEFTOVER detection is the stale one, so the newborn sits exactly on 17.0.
  EXPECT_NEAR(born.box().z, 17.0, 1e-12);
  // 1e-6, not 1e-12: this one has been through nine CT-UKF updates. See the tolerance note on
  // LifecycleStatePinFourModeBank below.
  EXPECT_NEAR(kept.box().z, 17.845764995098378, 1e-6);
}

// DEFECT KILLED: the IoU gate comparison inverted (`iou <= iou_gate` instead of `>=`). One
// detection with a FIXED overlap of 0.322 is offered to two trackers whose only difference is the
// gate, one below it and one above it. Inversion flips BOTH outcomes, so neither half can pass by
// accident. Python (verified): iou_3d(car(0,20), car(2,20)) = 0.32203389830508516;
// gate 0.01 -> 1 track (matched, hits 4); gate 0.9 -> 2 tracks (id 0 hits 3 tsu 1, id 1 hits 1).
TEST(Tracker, IouGateIsAFloorOnOverlapNotACeiling) {
  {
    TrackerConfig cfg;
    cfg.iou_gate = 0.01;                       // 0.322 is ABOVE the gate -> must match
    KittiTracker trk(cfg);
    for (int i = 0; i < 3; ++i) trk.step({car(0.0, 20.0)});
    trk.step({car(2.0, 20.0)});
    ASSERT_EQ(trk.numTracks(), 1u);
    EXPECT_EQ(trk.tracks()[0].id, 0);
    EXPECT_EQ(trk.tracks()[0].hits, 4u);
    EXPECT_EQ(trk.tracks()[0].time_since_update, 0u);
  }
  {
    TrackerConfig cfg;
    cfg.iou_gate = 0.9;                        // 0.322 is BELOW the gate -> must not match
    KittiTracker trk(cfg);
    for (int i = 0; i < 3; ++i) trk.step({car(0.0, 20.0)});
    trk.step({car(2.0, 20.0)});
    ASSERT_EQ(trk.numTracks(), 2u);
    EXPECT_EQ(trk.tracks()[0].id, 0);
    EXPECT_EQ(trk.tracks()[0].hits, 3u);
    EXPECT_EQ(trk.tracks()[0].time_since_update, 1u);
    EXPECT_EQ(trk.tracks()[1].id, 1);
    EXPECT_EQ(trk.tracks()[1].hits, 1u);
  }
}

// DEFECT KILLED: cost stored as `iou` instead of `1 - iou`. The solver MINIMIZES, so dropping the
// complement makes it prefer the WORST overlap. Two cars 3.0 m apart in x (a 3.9 m box, so every
// cross pair has IoU 0.130 and is gated IN) are re-detected in place; the correct pairing is the
// identity and the inverted-cost pairing is the swap.
//
// The discriminator is `score`, which is carried verbatim from the matched detection and has ZERO
// effect on IoU — so this reads the association decision directly, with no filter noise and no
// tolerance. Python (verified): after the second frame, id 0 -> score 0.25, id 1 -> score 0.75.
TEST(Tracker, CostIsOneMinusIouSoTheSolverPrefersMoreOverlap) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  trk.step({car(0.0, 20.0, 1.0), car(3.0, 20.0, 0.5)});
  ASSERT_EQ(trk.numTracks(), 2u);
  ASSERT_EQ(trk.tracks()[0].id, 0);
  ASSERT_NEAR(trk.tracks()[0].box().x, 0.0, 1e-15);
  ASSERT_NEAR(trk.tracks()[1].box().x, 3.0, 1e-15);

  trk.step({car(0.0, 20.0, 0.25), car(3.0, 20.0, 0.75)});
  ASSERT_EQ(trk.numTracks(), 2u);              // a swap would still give 2 tracks: check scores
  EXPECT_EQ(trk.tracks()[0].id, 0);
  EXPECT_EQ(trk.tracks()[1].id, 1);
  EXPECT_NEAR(trk.tracks()[0].box().score, 0.25, 1e-15);
  EXPECT_NEAR(trk.tracks()[1].box().score, 0.75, 1e-15);
}

// DEFECT KILLED: cfg.greedy not plumbed into associateFromCost (either branch hardwired).
//
// A greedy trap built out of real 3D-IoU. Tracks are born at x = 0.0 and x = 2.1 with zero
// velocity, so their PREDICTED boxes next frame sit exactly there; detections at x = 1.0 and
// x = -1.1 then give (Python, verified) cost
//     [[0.40816326530612246, 0.44               ],
//      [0.44,                0.9014084507042254]]
// whose global minimum (0,0) blocks the optimum. Hungarian pairs (0,1),(1,0) for 0.88; greedy
// commits to (0,0) and is forced into (1,1) for 1.31. `score` again carries the decision out.
TEST(Tracker, GreedyFlagChangesTheAssignment) {
  const auto run = [](bool greedy) {
    TrackerConfig cfg;
    cfg.greedy = greedy;
    KittiTracker trk(cfg);
    trk.step({car(0.0, 20.0, 1.0), car(2.1, 20.0, 1.0)});
    trk.step({car(1.0, 20.0, 0.25), car(-1.1, 20.0, 0.75)});
    return std::vector<double>{trk.tracks()[0].box().score, trk.tracks()[1].box().score};
  };

  const std::vector<double> hungarian = run(false);
  ASSERT_EQ(hungarian.size(), 2u);
  EXPECT_NEAR(hungarian[0], 0.75, 1e-15);      // id 0 took the FAR detection
  EXPECT_NEAR(hungarian[1], 0.25, 1e-15);

  const std::vector<double> greedy = run(true);
  ASSERT_EQ(greedy.size(), 2u);
  EXPECT_NEAR(greedy[0], 0.25, 1e-15);         // id 0 grabbed the locally cheapest one
  EXPECT_NEAR(greedy[1], 0.75, 1e-15);
}

// DEFECT KILLED: the Mahalanobis branch never taken, or its gate direction flipped. Unlike the
// IoU gate this one is a CEILING (`d2 <= kMahaGate`), so a copy-paste of the IoU comparison here
// would reject every plausible match. Python (verified): four frames of a target moving 0.6 m per
// frame stay on one confirmed track under cost="maha"; a detection at x = 60 is then far outside
// chi2.ppf(0.99, 2) = 9.21 (S ~ diag(0.388, 0.388)) and must birth a second track instead.
TEST(Tracker, MahalanobisCostModeAssociatesAndGates) {
  TrackerConfig cfg;
  cfg.cost = "maha";
  KittiTracker trk(cfg);
  std::vector<const BoxTrack*> out;
  for (int k = 0; k < 4; ++k) out = trk.step({car(0.0, 20.0 + 0.6 * k)});
  ASSERT_EQ(trk.numTracks(), 1u);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(trk.tracks()[0].hits, 4u);

  Vector2d z_pred;
  Matrix2d s;
  trk.tracks()[0].predictedMeasurement(z_pred, s);
  Vector2d far;
  far << 60.0 - z_pred(0), 20.0 - z_pred(1);
  EXPECT_GT(mahalanobisSq(far, s), 9.21);      // the detection below really is outside the gate

  trk.step({car(60.0, 20.0)});
  ASSERT_EQ(trk.numTracks(), 2u);
  EXPECT_EQ(trk.tracks()[0].id, 0);
  EXPECT_EQ(trk.tracks()[0].hits, 4u);
  EXPECT_EQ(trk.tracks()[0].time_since_update, 1u);
  EXPECT_EQ(trk.tracks()[1].id, 1);
  EXPECT_EQ(trk.tracks()[1].hits, 1u);
}

// ===========================================================================================
// tracker.hpp — output contract
// ===========================================================================================

TEST(Tracker, StepReturnsOnlyConfirmedTracksAndPointsIntoTracks) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  for (int i = 0; i < 3; ++i) trk.step({car(0.0, 10.0 + 0.5 * i)});
  const auto out = trk.step({car(0.0, 11.5), car(30.0, 40.0)});   // second detection is brand new

  ASSERT_EQ(trk.numTracks(), 2u);
  ASSERT_EQ(out.size(), 1u) << "a tentative track must not be returned";
  EXPECT_EQ(out[0]->id, 0);
  EXPECT_EQ(out[0], &trk.tracks()[0]) << "returned pointers must alias the live track storage";
  EXPECT_EQ(trk.tracks()[1].status, kT);
}

TEST(Tracker, CarriesLatestBoxGeometry) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  for (int i = 0; i < 3; ++i) trk.step({car(0.0, 10.0)});
  Box3D taller = car(0.0, 10.0);
  taller.h = 2.4;
  taller.yaw = 0.3;
  const auto out = trk.step({taller});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0]->box().h, 2.4, 1e-12);
  EXPECT_NEAR(out[0]->box().yaw, 0.3, 1e-12);
}

TEST(Tracker, BoxReturnsAFreshObjectEveryCall) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  trk.step({car(0.0, 10.0)});
  Box3D first = trk.tracks()[0].box();
  first.h = -999.0;                            // scribble on the returned value
  EXPECT_NEAR(first.h, -999.0, 1e-15);
  EXPECT_NEAR(trk.tracks()[0].box().h, 1.5, 1e-15);
}

TEST(Tracker, IdsAreStableAcrossAMovingTarget) {
  TrackerConfig cfg;
  cfg.min_hits = 3;
  KittiTracker trk(cfg);
  int id = -1;
  for (int k = 0; k < 12; ++k) {
    const auto out = trk.step({car(0.0, 10.0 + 0.5 * k)});
    ASSERT_EQ(trk.numTracks(), 1u) << "a spurious track appeared at step " << k;
    if (!out.empty()) {
      if (id < 0) id = out[0]->id;
      EXPECT_EQ(out[0]->id, id) << "id changed at step " << k;
    }
  }
  EXPECT_EQ(id, 0);
}

// ===========================================================================================
// tracker.hpp — the pinned 12-frame lifecycle
// ===========================================================================================

// The structural gate. Every counter of every track for every frame of scenarioFrame(), pinned to
// a run of the Python KittiTracker with default config (cost=iou, iou_gate=0.01, min_hits=3,
// max_age=2, greedy=false, p0_vel=10.0; imm dt=0.1, sigma_pos=0.5, q_accel=2.0,
// omegas=(0.2,-0.2), pi_diag=0.97).
//
// The interesting rows: id 1 coasts through frames 5 and 6 (tsu 1 then 2) and RE-ASSOCIATES at
// frame 7 keeping its id and continuing hits 5 -> 6; id 2 is born at frame 3, promotes at frame 5
// on its third hit, then dies at the EMPTY frame 8 when tsu reaches 3 > max_age.
TEST(Tracker, LifecyclePinMatchesPythonReference) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);

  const std::vector<std::vector<int>> want_confirmed{
      {}, {}, {0, 1}, {0, 1}, {0, 1}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1}, {0, 1}, {0, 1},
      {0, 1}};
  const std::vector<std::vector<TrackRow>> want_tracks{
      /* k=0  */ {{0, kT, 1, 1, 0, 0}, {1, kT, 1, 1, 0, 0}},
      /* k=1  */ {{0, kT, 2, 2, 0, 1}, {1, kT, 2, 2, 0, 1}},
      /* k=2  */ {{0, kC, 3, 3, 0, 2}, {1, kC, 3, 3, 0, 2}},
      /* k=3  */ {{0, kC, 4, 4, 0, 3}, {1, kC, 4, 4, 0, 3}, {2, kT, 1, 1, 0, 0}},
      /* k=4  */ {{0, kC, 5, 5, 0, 4}, {1, kC, 5, 5, 0, 4}, {2, kT, 2, 2, 0, 1}},
      /* k=5  */ {{0, kC, 6, 6, 0, 5}, {1, kC, 5, 0, 1, 5}, {2, kC, 3, 3, 0, 2}},
      /* k=6  */ {{0, kC, 7, 7, 0, 6}, {1, kC, 5, 0, 2, 6}, {2, kC, 3, 0, 1, 3}},
      /* k=7  */ {{0, kC, 8, 8, 0, 7}, {1, kC, 6, 1, 0, 7}, {2, kC, 3, 0, 2, 4}},
      /* k=8  */ {{0, kC, 8, 0, 1, 8}, {1, kC, 6, 0, 1, 8}},
      /* k=9  */ {{0, kC, 9, 1, 0, 9}, {1, kC, 7, 1, 0, 9}},
      /* k=10 */ {{0, kC, 10, 2, 0, 10}, {1, kC, 8, 2, 0, 10}},
      /* k=11 */ {{0, kC, 11, 3, 0, 11}, {1, kC, 9, 3, 0, 11}},
  };

  for (int k = 0; k < 12; ++k) {
    const auto out = trk.step(scenarioFrame(k));
    const std::vector<int>& want_ids = want_confirmed[static_cast<std::size_t>(k)];
    ASSERT_EQ(out.size(), want_ids.size()) << "confirmed count at frame " << k;
    for (std::size_t i = 0; i < want_ids.size(); ++i)
      EXPECT_EQ(out[i]->id, want_ids[i]) << "confirmed slot " << i << " at frame " << k;
    ASSERT_NO_FATAL_FAILURE(expectRows(trk, want_tracks[static_cast<std::size_t>(k)], k));
  }
}

// The same scenario on a CV+CA-only bank (omegas empty). This drops the two CT UKF modes but runs
// the entire tracker — cost matrix, association, lifecycle, IMM mixing — so it pins the numbers
// with the UKF conditioning removed and can therefore hold a tight tolerance.
//
// TOLERANCE 1e-9: test_imm.cpp measured the two-mode bank as bit-identical to numpy for mu/P and
// within 1 ulp for x on a single step; this accumulates twelve steps of a contractive linear
// filter, with no ill-conditioned weighted mean anywhere in the loop. If this one starts failing
// the bug is real. Values from the Python KittiTracker with IMMConfig(dt=0.1, sigma_pos=0.5,
// q_accel=2.0, omegas=()).
TEST(Tracker, LifecycleStatePinCvCaBank) {
  TrackerConfig cfg;
  cfg.imm.omegas.clear();
  KittiTracker trk(cfg);

  for (int k = 0; k < 12; ++k) {
    trk.step(scenarioFrame(k));
    Vector4d x0;
    Matrix4d p0;
    if (k == 0) {
      ASSERT_EQ(trk.numTracks(), 2u);
      ASSERT_EQ(trk.tracks()[0].modeProbabilities().size(), 2);
      trk.tracks()[0].immState(x0, p0);
      EXPECT_NEAR(x0(0), 0.0, 1e-12);
      EXPECT_NEAR(x0(1), 10.0, 1e-12);
      EXPECT_NEAR(p0(0, 0), 0.25, 1e-12);
      EXPECT_NEAR(p0(2, 2), 10.0, 1e-12);
    } else if (k == 2) {
      ASSERT_EQ(trk.numTracks(), 2u);
      trk.tracks()[0].immState(x0, p0);
      EXPECT_NEAR(x0(0), 0.0, 1e-9);
      EXPECT_NEAR(x0(1), 10.86718236573338, 1e-9);
      EXPECT_NEAR(x0(3), 2.6753166680642266, 1e-9);
      EXPECT_NEAR(p0(0, 0), 0.13902335952268674, 1e-9);
      EXPECT_NEAR(p0(2, 2), 5.611033797647302, 1e-9);
      Vector4d x1;
      Matrix4d p1;
      trk.tracks()[1].immState(x1, p1);
      EXPECT_NEAR(x1(0), 8.0, 1e-9);
      EXPECT_NEAR(x1(1), 19.421878476345803, 1e-9);
      EXPECT_NEAR(x1(3), -1.783543540885223, 1e-9);
    } else if (k == 5) {
      ASSERT_EQ(trk.numTracks(), 3u);
      trk.tracks()[0].immState(x0, p0);
      EXPECT_NEAR(x0(1), 12.815443302667429, 1e-9);
      EXPECT_NEAR(x0(3), 5.272858294359352, 1e-9);
      Vector4d x1;
      Matrix4d p1;
      trk.tracks()[1].immState(x1, p1);            // coasting: predicted, never corrected
      EXPECT_EQ(trk.tracks()[1].time_since_update, 1u);
      EXPECT_NEAR(x1(1), 18.2372911008381, 1e-9);
      EXPECT_NEAR(x1(3), -3.2129483889361063, 1e-9);
      EXPECT_NEAR(p1(0, 0), 0.2321516724724536, 1e-9);
      Vector4d x2;
      Matrix4d p2;
      trk.tracks()[2].immState(x2, p2);
      EXPECT_NEAR(x2(0), -9.0, 1e-9);
      EXPECT_NEAR(x2(1), 16.222651934662277, 1e-9);
    } else if (k == 8) {
      ASSERT_EQ(trk.numTracks(), 2u);              // the empty frame killed id 2
      trk.tracks()[0].immState(x0, p0);
      EXPECT_NEAR(x0(1), 14.655977084900062, 1e-9);
      EXPECT_NEAR(x0(3), 5.691532483582562, 1e-9);
      Vector4d x1;
      Matrix4d p1;
      trk.tracks()[1].immState(x1, p1);
      EXPECT_NEAR(x1(1), 16.949207147233928, 1e-9);
      EXPECT_NEAR(x1(3), -3.720726481406749, 1e-9);
    } else if (k == 11) {
      ASSERT_EQ(trk.numTracks(), 2u);
      trk.tracks()[0].immState(x0, p0);
      EXPECT_NEAR(x0(1), 16.547500018325323, 1e-9);
      EXPECT_NEAR(x0(3), 5.92993301989002, 1e-9);
      EXPECT_NEAR(p0(0, 0), 0.08569523227368181, 1e-9);
      EXPECT_NEAR(p0(2, 2), 0.3468086567954965, 1e-9);
      Vector4d x1;
      Matrix4d p1;
      trk.tracks()[1].immState(x1, p1);
      EXPECT_NEAR(x1(1), 15.636533192726322, 1e-9);
      EXPECT_NEAR(x1(3), -3.9537544274220133, 1e-9);
    }
  }
}

// The same scenario on the DEFAULT four-mode bank, so the CT UKF modes are live.
//
// TOLERANCE 1e-6, and the looseness is measured, not guessed. Perturbing frame 0's detections by
// 1e-12 and re-running the Python tracker moves the frame-11 state by 6.3e-10 and the frame-2
// state by 1.0e-9 (scratch experiment, 2026-07-27) — an amplification of ~1e3, coming from the
// CT modes' unscented weighted mean, which is conditioned at ~2e6. test_imm.cpp measured the C++
// four-mode bank against numpy at 3.9e-11 after ONE step, so ~1e-8 is the honest expectation
// here and 1e-6 leaves two orders of margin. Do NOT tighten this to 1e-9 without re-measuring;
// do NOT loosen it further, because at 1e-3 an association error would start slipping through.
TEST(Tracker, LifecycleStatePinFourModeBank) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  for (int k = 0; k < 12; ++k) {
    trk.step(scenarioFrame(k));
    if (k == 2) {
      ASSERT_EQ(trk.numTracks(), 2u);
      EXPECT_NEAR(trk.tracks()[0].box().z, 10.867026842633676, 1e-6);
      EXPECT_NEAR(trk.tracks()[1].box().z, 19.42198214988708, 1e-6);
    } else if (k == 8) {
      ASSERT_EQ(trk.numTracks(), 2u);
      EXPECT_NEAR(trk.tracks()[0].box().z, 14.652429053569488, 1e-6);
      EXPECT_NEAR(trk.tracks()[1].box().z, 16.951775094271515, 1e-6);
    } else if (k == 11) {
      ASSERT_EQ(trk.numTracks(), 2u);
      EXPECT_NEAR(trk.tracks()[0].box().z, 16.542878668887088, 1e-6);
      EXPECT_NEAR(trk.tracks()[1].box().z, 15.63920062189687, 1e-6);
      const Eigen::VectorXd mu = trk.tracks()[0].modeProbabilities();
      ASSERT_EQ(mu.size(), 4);
      EXPECT_NEAR(mu.sum(), 1.0, 1e-12);
      EXPECT_GT(mu.minCoeff(), 0.0);
    }
  }
}

// ---------------------------------------------------------------------------
// Gap closures from the 2026-07-27 mutation sweep (34 mutants, 21 killed by the
// tests above). Each of these kills a mutant that the suite above did NOT catch:
//   * gate applied before instead of after the solve  (changed the pairing on
//     8/4000 random matrices; baseline matches scipy, the mutant does not)
//   * update() dropping the l / w / y geometry carry
//   * greedy cost-tie ordering (documented contract, previously unpinned)
//   * kMahaGate: the suite passed with it anywhere in [1.0, 100.0]
//   * min_hits == 1 (newborn must confirm on its birth frame)
//   * a TENTATIVE track missing a frame (promotion must count cumulative hits,
//     not hit_streak) -- no prior test ever let a tentative track miss
// ---------------------------------------------------------------------------
namespace {
Box3D gapCar(double x, double z, double score = 1.0) {
  Box3D b;
  b.x = x;
  b.y = 1.6;
  b.z = z;
  b.yaw = 0.0;
  b.l = 3.9;
  b.w = 1.6;
  b.h = 1.5;
  b.score = score;
  return b;
}
}  // namespace

// kills M15 (gate filter applied BEFORE the solve)
TEST(Gap, GateIsAppliedAfterTheSolveNotBefore) {
  const double B = kBigCost;
  {   // fully-gated COLUMN 0 plus a tie between rows 0 and 1 on column 1.
    Eigen::MatrixXd c(3, 2);
    c << B, 0.0,
         B, 0.0,
         B, 1.0;
    const Assignment a = associateFromCost(c, kBigCost, false);
    ASSERT_EQ(a.matches.size(), 1u);
    EXPECT_EQ(a.matches[0].first, 1);        // scipy: [(1, 1)], cols [0], rows [0, 2]
    EXPECT_EQ(a.matches[0].second, 1);
    ASSERT_EQ(a.unmatched_rows.size(), 2u);
    EXPECT_EQ(a.unmatched_rows[0], 0);
    EXPECT_EQ(a.unmatched_rows[1], 2);
  }
  {   // fully-gated ROW 2, degenerate optimum over rows 0/1.
    Eigen::MatrixXd c(4, 3);
    c << 2.0, 1.0, 1.0,
         3.0, 2.0, 1.0,
         B,   B,   B,
         B,   B,   3.0;
    const Assignment a = associateFromCost(c, kBigCost, false);
    ASSERT_EQ(a.matches.size(), 3u);         // scipy: [(0,0), (1,1), (3,2)]
    EXPECT_EQ(a.matches[0].second, 0);
    EXPECT_EQ(a.matches[1].second, 1);
    EXPECT_EQ(a.matches[2].second, 2);
  }
  {
    Eigen::MatrixXd c(3, 3);
    c << 2.0, 3.0, 0.0,
         B,   B,   B,
         B,   3.0, 1.0;
    const Assignment a = associateFromCost(c, kBigCost, false);
    ASSERT_EQ(a.matches.size(), 2u);         // scipy: [(0,2), (2,1)]
    EXPECT_EQ(a.matches[0].first, 0);
    EXPECT_EQ(a.matches[0].second, 2);
    EXPECT_EQ(a.matches[1].first, 2);
    EXPECT_EQ(a.matches[1].second, 1);
  }
}

// kills X04 (greedy tie-break order)
TEST(Gap, GreedyBreaksCostTiesInGenerationOrder) {
  Eigen::MatrixXd c(2, 2);
  c << kBigCost, 1.0,
       1.0, kBigCost;
  const Assignment g = associateFromCost(c, kBigCost, true);
  ASSERT_EQ(g.matches.size(), 2u);           // Python's stable sort emits (0,1) before (1,0)
  EXPECT_EQ(g.matches[0].first, 0);
  EXPECT_EQ(g.matches[0].second, 1);
  EXPECT_EQ(g.matches[1].first, 1);
  EXPECT_EQ(g.matches[1].second, 0);
}

// kills X02 and X03 (l/w/y not carried by update())
TEST(Gap, CarriesEveryBoxDimensionNotJustHeightAndYaw) {
  TrackerConfig cfg;
  KittiTracker trk(cfg);
  for (int i = 0; i < 3; ++i) trk.step({gapCar(0.0, 10.0)});
  Box3D grown = gapCar(0.0, 10.0);
  grown.l = 5.2;
  grown.w = 2.1;
  grown.h = 2.4;
  grown.y = 0.9;
  grown.yaw = 0.3;
  const auto out = trk.step({grown});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NEAR(out[0]->box().l, 5.2, 1e-12);
  EXPECT_NEAR(out[0]->box().w, 2.1, 1e-12);
  EXPECT_NEAR(out[0]->box().h, 2.4, 1e-12);
  EXPECT_NEAR(out[0]->box().y, 0.9, 1e-12);
  EXPECT_NEAR(out[0]->box().yaw, 0.3, 1e-12);
}

// kills X05 (kMahaGate loosened/tightened)
TEST(Gap, MahalanobisGateIsChiSquared99NotSomethingTighter) {
  EXPECT_NEAR(kMahaGate, 9.21, 1e-12);       // chi2.ppf(0.99, 2), matching Python's _MAHA_GATE

  TrackerConfig cfg;
  cfg.cost = "maha";
  KittiTracker trk(cfg);
  KittiTracker probe(cfg);
  for (int k = 0; k < 4; ++k) {
    trk.step({gapCar(0.0, 20.0 + 0.6 * k)});
    probe.step({gapCar(0.0, 20.0 + 0.6 * k)});
  }
  probe.step({});                            // predict + coast: probe now holds the state the
  ASSERT_EQ(probe.numTracks(), 1u);          // NEXT cost matrix of `trk` will be built from
  Vector2d z_pred;
  Matrix2d s;
  probe.tracks()[0].predictedMeasurement(z_pred, s);

  Vector2d u;
  u << 1.0, 0.0;
  const double alpha = std::sqrt(7.5 / mahalanobisSq(u, s));   // d2 == 7.5 exactly
  Vector2d d = alpha * u;
  ASSERT_NEAR(mahalanobisSq(d, s), 7.5, 1e-9);
  // 7.5 sits strictly between chi2.ppf(0.95, 2) = 5.99 and chi2.ppf(0.99, 2) = 9.21, so only the
  // correct gate accepts it.
  trk.step({gapCar(z_pred(0) + d(0), z_pred(1) + d(1))});
  ASSERT_EQ(trk.numTracks(), 1u) << "a detection inside chi2(0.99) must match, not birth";
  EXPECT_EQ(trk.tracks()[0].hits, 5u);
  EXPECT_EQ(trk.tracks()[0].time_since_update, 0u);
}

// kills X09 and X10 (promotion / erase moved ahead of the birth loop)
TEST(Gap, MinHitsOfOneConfirmsOnTheBirthFrame) {
  TrackerConfig cfg;
  cfg.min_hits = 1;
  KittiTracker trk(cfg);
  const auto out = trk.step({gapCar(0.0, 10.0)});
  ASSERT_EQ(trk.numTracks(), 1u);
  EXPECT_EQ(trk.tracks()[0].status, kC);
  ASSERT_EQ(out.size(), 1u) << "min_hits == 1 must confirm and publish on the birth frame";
  EXPECT_EQ(out[0]->id, 0);
}

// kills X11 (promotion counting hit_streak instead of cumulative hits)
TEST(Gap, TentativeTrackPromotesOnCumulativeHitsAcrossAMiss) {
  TrackerConfig cfg;
  cfg.min_hits = 3;
  cfg.max_age = 2;
  KittiTracker trk(cfg);
  EXPECT_TRUE(trk.step({gapCar(0.0, 10.0)}).empty());       // hit 1
  EXPECT_TRUE(trk.step({gapCar(0.0, 10.1)}).empty());       // hit 2
  EXPECT_TRUE(trk.step({}).empty());                     // miss while STILL TENTATIVE
  ASSERT_EQ(trk.numTracks(), 1u);
  EXPECT_EQ(trk.tracks()[0].hits, 2u);
  EXPECT_EQ(trk.tracks()[0].hit_streak, 0u);
  EXPECT_EQ(trk.tracks()[0].status, kT);

  const auto out = trk.step({gapCar(0.0, 10.2)});           // cumulative hit 3 -> confirm
  ASSERT_EQ(trk.numTracks(), 1u);
  EXPECT_EQ(trk.tracks()[0].hits, 3u);
  EXPECT_EQ(trk.tracks()[0].hit_streak, 1u);
  EXPECT_EQ(trk.tracks()[0].status, kC);
  ASSERT_EQ(out.size(), 1u) << "min_hits counts TOTAL hits, not consecutive ones";
}
