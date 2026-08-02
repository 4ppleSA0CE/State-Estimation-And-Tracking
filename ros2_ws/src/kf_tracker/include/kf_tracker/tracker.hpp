// KITTI Car tracker: BEV-center IMM + carried box geometry, 3D-IoU association (Mahalanobis
// behind a flag), M-of-N birth / max-age death.
//
// A 1:1 C++ port of prototypes/python/tracking/kitti_tracker.py. One deliberate structural
// divergence: p0_vel lives ONLY on TrackerConfig. The Python has it on both IMMConfig and
// KittiTrackerConfig with the tracker one silently shadowing the other, which is a trap, not a
// feature (see L7 in the C++ tracker port plan). No numeric effect.
#ifndef KF_TRACKER_TRACKER_HPP
#define KF_TRACKER_TRACKER_HPP

#include <algorithm>   // std::remove_if
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "kf_tracker/association.hpp"
#include "kf_tracker/box3d.hpp"
#include "kf_tracker/imm.hpp"

namespace kf_tracker {

enum class TrackStatus { kTentative, kConfirmed, kDead };

inline constexpr double kMahaGate = 9.21;   // chi2.ppf(0.99, 2)

// 10 Hz KITTI defaults, mirroring kitti_tracker._default_imm(): sigma_pos ~ detection-center
// noise (m), q_accel for road vehicles, a +/- CT bank for turns. Built field by field rather than
// by aggregate initialization so that adding a field to ImmConfig cannot silently shift these.
inline ImmConfig defaultTrackerImm() {
  ImmConfig c;
  c.dt = 0.1;
  c.sigma_pos = 0.5;
  c.q_accel = 2.0;
  c.omegas = {0.2, -0.2};
  c.pi_diag = 0.97;
  return c;
}

struct TrackerConfig {
  ImmConfig imm = defaultTrackerImm();
  std::string cost = "iou";     // "iou" | "maha"
  double iou_gate = 0.01;       // minimum IoU to allow a match (AB3DMOT's setting)
  int min_hits = 3;
  int max_age = 2;
  bool greedy = false;
  double p0_vel = 10.0;         // birth velocity-uncertainty
};

class BoxTrack {
 public:
  BoxTrack(int track_id, const Box3D& det, const TrackerConfig& cfg)
      : id(track_id),
        imm_(cfg.imm, Matrix2d::Identity() * (cfg.imm.sigma_pos * cfg.imm.sigma_pos)),
        y_(det.y), l_(det.l), w_(det.w), h_(det.h), yaw_(det.yaw), score_(det.score) {
    Vector4d x;
    x << det.x, det.z, 0.0, 0.0;               // position known, velocity unknown
    Matrix4d p = Matrix4d::Zero();
    p.diagonal() << cfg.imm.sigma_pos * cfg.imm.sigma_pos,
                    cfg.imm.sigma_pos * cfg.imm.sigma_pos,
                    cfg.p0_vel, cfg.p0_vel;
    imm_.initState(x, p);
  }

  int id;
  TrackStatus status = TrackStatus::kTentative;
  unsigned hits = 1;
  unsigned hit_streak = 1;
  unsigned time_since_update = 0;
  unsigned age = 0;

  void predict() {
    imm_.predict();
    ++age;
  }

  void update(const Box3D& det) {
    Vector2d z;
    z << det.x, det.z;
    imm_.update(z);
    y_ = det.y;                 // carry the latest geometry — only the BEV centre is filtered
    l_ = det.l;
    w_ = det.w;
    h_ = det.h;
    yaw_ = det.yaw;
    score_ = det.score;
    ++hits;
    ++hit_streak;
    time_since_update = 0;      // reset on UPDATE; markMissed() is what increments it
  }

  void markMissed() {
    imm_.coast();               // intentionally a no-op: predict() already advanced every mode
    hit_streak = 0;
    ++time_since_update;
  }

  // A FRESH Box3D every call — never a reference into mutable filter state. Snapshotting the
  // returned value at step time is what a consumer must do (L5 in the C++ tracker port plan).
  Box3D box() const {
    Vector4d x;
    Matrix4d p;
    imm_.state(x, p);
    Box3D b;
    b.x = x(0);
    b.y = y_;
    b.z = x(1);
    b.yaw = yaw_;
    b.l = l_;
    b.w = w_;
    b.h = h_;
    b.score = score_;
    b.track_id = id;
    return b;
  }

  // Same estimate as box(). Distinct name because the ONLY caller is the cost matrix, which reads
  // it after every track has been predicted and before any has been updated — mirrors Python's
  // predicted_box(), which is likewise just box() under a name that documents the call site.
  Box3D predictedBox() const { return box(); }

  void predictedMeasurement(Vector2d& z_pred, Matrix2d& s) const {
    imm_.predictedMeasurement(z_pred, s);
  }

  void immState(Vector4d& x, Matrix4d& p) const { imm_.state(x, p); }
  const Eigen::VectorXd& modeProbabilities() const { return imm_.modeProbabilities(); }

 private:
  ImmFilter imm_;
  double y_, l_, w_, h_, yaw_, score_;
};

class KittiTracker {
 public:
  explicit KittiTracker(const TrackerConfig& cfg = TrackerConfig()) : cfg_(cfg) {}

  std::size_t numTracks() const { return tracks_.size(); }
  const std::vector<BoxTrack>& tracks() const { return tracks_; }

  // One detection frame -> pointers to the confirmed tracks after this frame.
  //
  // Pointers, not copies: BoxTrack owns an ImmFilter, which owns unique_ptrs to the mode bank, so
  // BoxTrack is move-only and cannot be returned by value in a vector. Pointers also make the
  // lifetime hazard explicit — THEY ARE INVALIDATED BY THE NEXT step() (tracks_ is erased from,
  // appended to, and may reallocate). A caller that wants per-frame results must copy
  // id / box() / immState() / modeProbabilities() immediately; holding the pointers and reading
  // them later yields every track's FINAL state for every frame. That is the Python evaluator's
  // bug, and it cost a full debugging cycle.
  //
  // STEP ORDER IS LOAD-BEARING and must stay identical to KittiTracker.step in the Python:
  //   predict ALL tracks -> build cost -> associate -> update matched -> markMissed unmatched
  //   -> birth from unmatched detections -> promote/kill -> erase dead -> return confirmed.
  // Building the cost matrix before the predict silently shifts every association by one frame
  // (the association is scored against last frame's posterior instead of this frame's prior) and
  // leaves every state value unchanged, so only an association-outcome test can see it.
  //
  // An EMPTY detection frame still steps the tracker: everything predicts, coasts and ages. Do
  // not add an early return on detections.empty() — the reference .npz has deliberately blanked
  // frames that exercise this path.
  std::vector<const BoxTrack*> step(const std::vector<Box3D>& detections) {
    for (auto& t : tracks_) t.predict();

    Assignment assign;
    if (!tracks_.empty() && !detections.empty()) {
      assign = associateFromCost(costMatrix(detections), kBigCost, cfg_.greedy);
    } else {
      for (std::size_t j = 0; j < detections.size(); ++j)
        assign.unmatched_cols.push_back(static_cast<int>(j));
      for (std::size_t i = 0; i < tracks_.size(); ++i)
        assign.unmatched_rows.push_back(static_cast<int>(i));
    }

    for (const auto& m : assign.matches)
      tracks_[static_cast<std::size_t>(m.first)].update(detections[static_cast<std::size_t>(m.second)]);
    for (int i : assign.unmatched_rows) tracks_[static_cast<std::size_t>(i)].markMissed();
    // Births go LAST so the new tracks are not predicted, associated or marked missed this frame.
    // They are appended, so the row indices above stay valid either way; what actually matters is
    // that birth happens after the predict loop.
    for (int j : assign.unmatched_cols)
      tracks_.emplace_back(next_id_++, detections[static_cast<std::size_t>(j)], cfg_);

    for (auto& t : tracks_) {
      // >= min_hits: a track promotes ON its min_hits-th hit, not one frame later.
      if (t.status == TrackStatus::kTentative && t.hits >= static_cast<unsigned>(cfg_.min_hits))
        t.status = TrackStatus::kConfirmed;
      // > max_age: max_age consecutive misses are SURVIVED; the (max_age + 1)-th kills.
      if (t.time_since_update > static_cast<unsigned>(cfg_.max_age)) t.status = TrackStatus::kDead;
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const BoxTrack& t) { return t.status == TrackStatus::kDead; }),
                  tracks_.end());

    std::vector<const BoxTrack*> confirmed;
    for (const auto& t : tracks_)
      if (t.status == TrackStatus::kConfirmed) confirmed.push_back(&t);
    return confirmed;
  }

 private:
  // Rows are tracks, columns are detections; gated-out pairs keep the kBigCost fill.
  Eigen::MatrixXd costMatrix(const std::vector<Box3D>& dets) const {
    const int n = static_cast<int>(tracks_.size());
    const int m = static_cast<int>(dets.size());
    Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(n, m, kBigCost);

    if (cfg_.cost == "iou") {
      for (int i = 0; i < n; ++i) {
        const Box3D pb = tracks_[static_cast<std::size_t>(i)].predictedBox();
        for (int j = 0; j < m; ++j) {
          const double iou = iou3d(pb, dets[static_cast<std::size_t>(j)]);
          // Gate is a FLOOR on overlap: only pairs with at least iou_gate overlap are allowed,
          // and the cost is 1 - iou so the solver (a MINIMIZER) prefers more overlap.
          if (iou >= cfg_.iou_gate) cost(i, j) = 1.0 - iou;
        }
      }
    } else {   // Mahalanobis on BEV centres
      for (int i = 0; i < n; ++i) {
        Vector2d z_pred;
        Matrix2d s;
        tracks_[static_cast<std::size_t>(i)].predictedMeasurement(z_pred, s);
        for (int j = 0; j < m; ++j) {
          const Box3D& db = dets[static_cast<std::size_t>(j)];
          Vector2d d;
          d << db.x - z_pred(0), db.z - z_pred(1);
          const double d2 = mahalanobisSq(d, s);
          // Gate is a CEILING on distance here — the opposite sense to the IoU branch above.
          if (d2 <= kMahaGate) cost(i, j) = d2;
        }
      }
    }
    return cost;
  }

  TrackerConfig cfg_;
  std::vector<BoxTrack> tracks_;
  int next_id_ = 0;
};

}  // namespace kf_tracker

#endif  // KF_TRACKER_TRACKER_HPP
