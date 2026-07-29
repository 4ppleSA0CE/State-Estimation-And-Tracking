// Detection coupling kernel: base_link box + estimated ego pose -> map_bev box, plus the
// stamp-keyed ego-pose store the ROS node needs to pair the two deterministically.
//
// Frames (design doc section 4):
//   base_link  x forward, y left,  z up      -- boxes as published by pipeline_replay
//   ENU map    x East,    y North, z Up      -- the ESKF's frame
//   map_bev    x East,    y DOWN,  z North   -- ENU written in the KITTI-camera convention
//               that kf_tracker/box3d.hpp already implements, so the tracking kernel is
//               reused with zero edits.
//
// Box3D::y therefore means "left" on the way IN and "down" on the way OUT. That is deliberate:
// the struct is a frame-agnostic container, and the frame is carried by the message header.
//
// pose.position is the box BOTTOM-CENTER in every frame (the KITTI label convention), which is
// why the height mapping is a plain negation with no h/2 term anywhere.
//
// ROS-independent: Eigen + standard library only, so the gtest host-compiles.
#ifndef KF_TRACKER_EGO_TRANSFORM_HPP
#define KF_TRACKER_EGO_TRANSFORM_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>

#include <Eigen/Dense>

#include "kf_tracker/box3d.hpp"

namespace kf_tracker {

struct EgoPose {
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
};

// Heading = yaw of the rotation about ENU up. Taken from the rotation matrix rather than from
// an Euler decomposition so a composed roll/pitch/yaw still yields the heading (see the
// HeadingComesFromTheRotation test).
inline double egoHeading(const EgoPose& ego) {
  const Eigen::Matrix3d r = ego.q.toRotationMatrix();
  return std::atan2(r(1, 0), r(0, 0));
}

// base_link -> map_bev. Position uses the FULL rotation so roll and pitch are exact; box yaw
// composes with the heading only, because a BEV box yaw has no roll/pitch meaning. On
// drive_0001 roll and pitch stay under ~2 deg, so the induced box-yaw error is far below the
// detection noise -- stated in the design doc rather than hidden here.
inline Box3D transformToMapBev(const Box3D& box_body, const EgoPose& ego) {
  const Eigen::Vector3d p_body(box_body.x, box_body.y, box_body.z);
  const Eigen::Vector3d p_enu = ego.q * p_body + ego.p;
  const double yaw_enu = box_body.yaw + egoHeading(ego);

  Box3D out = box_body;      // l, w, h, score, track_id carry through untouched
  out.x = p_enu.x();         // East
  out.y = -p_enu.z();        // Down
  out.z = p_enu.y();         // North
  out.yaw = -yaw_enu;        // a rotation about down is the negated rotation about up
  return out;
}

// Fixed-capacity stamp -> pose store, ordered by stamp so eviction drops the OLDEST rather
// than the least-recently-inserted (ROS delivery is not ordered across topics).
//
// find() is EXACT-match only, by design. Nearest-match would make the transformed output
// depend on which messages happened to have arrived, which is the exact nondeterminism this
// class exists to remove.
class EgoPoseBuffer {
 public:
  explicit EgoPoseBuffer(std::size_t capacity = 512)
      : capacity_(capacity == 0 ? 1 : capacity) {}

  void insert(std::int64_t stamp_ns, const EgoPose& pose) {
    poses_[stamp_ns] = pose;
    while (poses_.size() > capacity_) poses_.erase(poses_.begin());
  }

  const EgoPose* find(std::int64_t stamp_ns) const {
    const auto it = poses_.find(stamp_ns);
    return it == poses_.end() ? nullptr : &it->second;
  }

  // Sentinel on an empty buffer so a "detection older than anything we hold" test cannot
  // accidentally succeed against a stale value.
  std::int64_t oldestStamp() const {
    return poses_.empty() ? std::numeric_limits<std::int64_t>::min() : poses_.begin()->first;
  }

  bool full() const { return poses_.size() >= capacity_; }
  std::size_t size() const { return poses_.size(); }

 private:
  std::size_t capacity_;
  std::map<std::int64_t, EgoPose> poses_;
};

}  // namespace kf_tracker

#endif  // KF_TRACKER_EGO_TRANSFORM_HPP
