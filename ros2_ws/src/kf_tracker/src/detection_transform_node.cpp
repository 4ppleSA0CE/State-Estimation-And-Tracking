// Stage 6 coupling node: base_link detections + the ESKF's estimated ego pose -> map_bev
// detections. This is the step that makes localization error show up in object positions.
//
// ORDERING, which is the whole reason this node is not three lines:
//   eskf_node publishes /ego/state for stamp t_k only when IMU t_{k+1} arrives (it emits the
//   pending step first, then predicts -- eskf_node.cpp:131). So when the detection frame for
//   t_k is published, the matching ego state does not exist yet. On top of that the two topics
//   reach this node over different DDS paths, so no publish ordering upstream would be a
//   delivery guarantee. "Use the latest ego state" is therefore both wrong and, worse,
//   nondeterministic run to run.
//
//   Rule: a detection frame publishes only when its EXACT stamp is present in the ego buffer.
//   Frames wait in a bounded queue; each arriving EgoState drains whatever it unblocks. Output
//   consequently lags input by one IMU period (10 ms), which every consumer absorbs because
//   they key off the detection stamp, not arrival time.
//
//   That queue lives in kf_tracker/pending_frames.hpp, NOT here: its behaviour is a function of
//   arrival order, which a ROS test can only produce by racing two DDS topics. Extracted, the
//   orders that matter are replayed in-process by test_pending_frames.cpp. What remains in this
//   file is message marshalling -- decode, hand to the queue, transform, publish.
//
// Exception policy mirrors tracker_node: rclcpp does not catch exceptions out of a
// subscription callback, so the callback bodies are wrapped. Unlike tracker_node this node
// publishes NOTHING on failure -- /detections_map has no 1:1-with-input contract to keep, and
// handing the tracker a silently-empty frame would age every track as if the scene had gone
// dark, which is strictly more misleading than a gap.
//
// THREADING: pending_ and ego_buffer_ are unguarded, and are correct only because both
// callbacks run on the one thread of rclcpp::spin(). Swapping in a MultiThreadedExecutor (or
// putting the two subscriptions in reentrant callback groups) makes this silently racy rather
// than loudly broken -- concurrent onEgo/onDetections would interleave a std::map insert with a
// deque drain. Any such change must come with a mutex around both members.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <kf_msgs/msg/detection.hpp>
#include <kf_msgs/msg/detection_array.hpp>
#include <kf_msgs/msg/ego_state.hpp>

#include <kf_tracker/box3d.hpp>
#include <kf_tracker/ego_transform.hpp>
#include <kf_tracker/pending_frames.hpp>

namespace kf_tracker {
namespace {

std::int64_t stampNs(const builtin_interfaces::msg::Time& t) {
  return static_cast<std::int64_t>(t.sec) * 1000000000LL + static_cast<std::int64_t>(t.nanosec);
}

// Yaw about the body Z axis -- this node's INPUT is base_link, where z is UP.
//
// This is deliberately NOT tracker_node's yawFromQuaternion, which decodes yaw about Y because
// the boxes it receives are already in map_bev (y down, KITTI-camera convention). The two are
// one character apart to read and completely different in effect: feeding a pure base_link
// heading quaternion (w, 0, 0, sin) to the Y-axis formula returns 0.0 for EVERY heading, so
// every box yaw would silently collapse to zero and only show up as degraded IoU association.
// Do not unify these two helpers.
//
//   yaw about z = atan2(2(wz + xy), 1 - 2(y^2 + z^2))  == atan2(R(1,0), R(0,0))   <- here
//   yaw about y = atan2(2(wy + xz), 1 - 2(y^2 + x^2))  == atan2(R(0,2), R(2,2))   <- tracker_node
double yawAboutZFromQuaternion(const geometry_msgs::msg::Quaternion& q) {
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

}  // namespace

class DetectionTransformNode : public rclcpp::Node {
 public:
  DetectionTransformNode() : Node("detection_transform_node") {
    // ROS 2 has exactly one integral parameter type (int64), so both counts are declared as
    // int64 and validated BEFORE narrowing: static_cast<std::size_t>(-1) is 2^64-1, not 0, so a
    // negative override would sail past a post-cast `== 0` check and configure an unbounded
    // queue instead of failing.
    const std::int64_t capacity_param = declare_parameter<std::int64_t>("ego_buffer_capacity", 512);
    const std::int64_t pending_param = declare_parameter<std::int64_t>("max_pending_frames", 32);
    map_bev_frame_ = declare_parameter<std::string>("map_bev_frame", "map_bev");

    if (pending_param < 1)
      throw std::invalid_argument("max_pending_frames must be >= 1, got " +
                                  std::to_string(pending_param));
    if (capacity_param < 1)
      throw std::invalid_argument("ego_buffer_capacity must be >= 1, got " +
                                  std::to_string(capacity_param));

    const auto capacity = static_cast<std::size_t>(capacity_param);
    max_pending_ = static_cast<std::size_t>(pending_param);
    ego_buffer_ = std::make_unique<EgoPoseBuffer>(capacity);
    pending_ = std::make_unique<PendingQueue>(max_pending_);
    pending_->setDropObserver([this](const DroppedFrame& drop) { logDrop(drop); });

    auto qos = rclcpp::QoS(rclcpp::KeepLast(2000)).reliable();
    out_pub_ = create_publisher<kf_msgs::msg::DetectionArray>("/detections_map", qos);
    ego_sub_ = create_subscription<kf_msgs::msg::EgoState>(
        "/ego/state", qos,
        [this](kf_msgs::msg::EgoState::ConstSharedPtr m) { onEgo(std::move(m)); });
    det_sub_ = create_subscription<kf_msgs::msg::DetectionArray>(
        "/detections", qos,
        [this](kf_msgs::msg::DetectionArray::ConstSharedPtr m) { onDetections(std::move(m)); });

    RCLCPP_INFO(get_logger(),
                "DetectionTransformNode ready (ego_buffer_capacity=%zu, max_pending_frames=%zu, "
                "map_bev_frame=%s)",
                capacity, max_pending_, map_bev_frame_.c_str());
  }

  // The drop counters are otherwise visible only through 1 Hz-throttled ERROR lines, so a fast
  // replay can lose a hundred frames and print one log line per wall-second. Say the totals
  // once, unconditionally, on the way out.
  ~DetectionTransformNode() override {
    RCLCPP_INFO(get_logger(),
                "DetectionTransformNode shutting down: published=%llu frames, dropped=%llu, "
                "%zu still pending",
                static_cast<unsigned long long>(published_),
                static_cast<unsigned long long>(pending_ ? pending_->dropped() : 0),
                pending_ ? pending_->size() : 0);
  }

 private:
  using PendingQueue = PendingFrameQueue<kf_msgs::msg::DetectionArray::ConstSharedPtr>;

  void onEgo(kf_msgs::msg::EgoState::ConstSharedPtr msg) {
    try {
      EgoPose pose;
      pose.p = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
      pose.q = Eigen::Quaterniond(msg->pose.orientation.w, msg->pose.orientation.x,
                                  msg->pose.orientation.y, msg->pose.orientation.z);
      // Eigen's normalize() on a zero or non-finite quaternion divides by ~0 and yields NaN
      // silently, which would then poison every transformed detection from here on. Reject it
      // loudly instead -- the catch below turns this into a throttled error, not a crash.
      const double qn = pose.q.norm();
      if (!std::isfinite(qn) || qn < 1e-9)
        throw std::invalid_argument("EgoState carries a degenerate orientation quaternion");
      pose.q.normalize();
      ego_buffer_->insert(stampNs(msg->header.stamp), pose);
      pending_->onEgo(*ego_buffer_, emits_);
      publishEmitted();
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), clock_, 1000, "ego callback failed: %s", e.what());
    }
  }

  void onDetections(kf_msgs::msg::DetectionArray::ConstSharedPtr msg) {
    try {
      const std::int64_t stamp = stampNs(msg->header.stamp);
      pending_->onFrame(stamp, std::move(msg), *ego_buffer_, emits_);
      publishEmitted();
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), clock_, 1000, "detection callback failed: %s", e.what());
    }
  }

  // Everything the queue released, oldest first. emits_ is a reused buffer, cleared by the
  // queue on entry and released here so the message shared_ptrs do not outlive the callback.
  void publishEmitted() {
    for (const auto& e : emits_) publishTransformed(*e.frame, e.ego);
    emits_.clear();
  }

  // The queue counts drops; this turns them into the diagnostics they used to be inline. Still
  // throttled -- a stream of them is one fault, not N -- with the totals repeated in the
  // destructor so a throttled burst is never the only record.
  void logDrop(const DroppedFrame& drop) {
    if (drop.reason == DropReason::kQueueOverflow) {
      RCLCPP_ERROR_THROTTLE(get_logger(), clock_, 1000,
                            "pending queue full (%zu); dropping the oldest detection frame "
                            "(published=%llu, total dropped=%llu)",
                            max_pending_, static_cast<unsigned long long>(published_),
                            static_cast<unsigned long long>(pending_->dropped()));
      return;
    }
    RCLCPP_ERROR_THROTTLE(get_logger(), clock_, 1000,
                          "detection frame at stamp %lld is older than the ego buffer "
                          "(oldest=%lld); it can never be matched (published=%llu, "
                          "total dropped=%llu)",
                          static_cast<long long>(drop.stamp),
                          static_cast<long long>(drop.ego_oldest_stamp),
                          static_cast<unsigned long long>(published_),
                          static_cast<unsigned long long>(pending_->dropped()));
  }

  void publishTransformed(const kf_msgs::msg::DetectionArray& in, const EgoPose& ego) {
    kf_msgs::msg::DetectionArray out;
    out.header.stamp = in.header.stamp;
    out.header.frame_id = map_bev_frame_;
    out.detections.reserve(in.detections.size());

    for (const auto& d : in.detections) {
      Box3D b;
      b.x = d.pose.position.x;
      b.y = d.pose.position.y;
      b.z = d.pose.position.z;
      b.yaw = yawAboutZFromQuaternion(d.pose.orientation);   // base_link: z is up
      b.l = d.dimensions.x;
      b.w = d.dimensions.y;
      b.h = d.dimensions.z;

      const Box3D m = transformToMapBev(b, ego);

      // NOTE: covariance passes through UN-ROTATED. This message's header says map_bev, but
      // o.covariance is still the 6x6 the detector emitted in base_link; rotating it properly
      // means R*C*R^T with the same base_link->map_bev basis change applied blockwise.
      // Inert today -- tracker_node never reads Detection.covariance, it uses its own
      // measurement noise -- so nothing observable changes either way. It is a landmine for
      // the first consumer that DOES trust it: fix it here before any such consumer ships.
      kf_msgs::msg::Detection o = d;      // carries covariance, classification, object_id
      o.pose.position.x = m.x;
      o.pose.position.y = m.y;
      o.pose.position.z = m.z;
      // Pure yaw about the (now downward) Y axis -- the exact inverse of the yaw decoding
      // tracker_node performs, which is the consumer of this topic.
      o.pose.orientation.w = std::cos(m.yaw * 0.5);
      o.pose.orientation.x = 0.0;
      o.pose.orientation.y = std::sin(m.yaw * 0.5);
      o.pose.orientation.z = 0.0;
      out.detections.push_back(o);
    }

    try {
      out_pub_->publish(out);
      ++published_;
    } catch (const std::exception& e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), clock_, 1000, "publishing /detections_map failed: %s",
                            e.what());
    }
  }

  std::unique_ptr<EgoPoseBuffer> ego_buffer_;
  std::unique_ptr<PendingQueue> pending_;
  std::vector<PendingQueue::Emit> emits_;
  std::size_t max_pending_ = 32;
  std::string map_bev_frame_ = "map_bev";
  std::uint64_t published_ = 0;
  // Steady clock for log throttling: a ROS-time clock would stall the throttle window under
  // use_sim_time and suppress every error after the first.
  rclcpp::Clock clock_{RCL_STEADY_TIME};

  rclcpp::Publisher<kf_msgs::msg::DetectionArray>::SharedPtr out_pub_;
  rclcpp::Subscription<kf_msgs::msg::EgoState>::SharedPtr ego_sub_;
  rclcpp::Subscription<kf_msgs::msg::DetectionArray>::SharedPtr det_sub_;
};

}  // namespace kf_tracker

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  int exit_code = 0;
  try {
    rclcpp::spin(std::make_shared<kf_tracker::DetectionTransformNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("detection_transform_node"), "aborting: %s", e.what());
    exit_code = 1;
  }
  rclcpp::shutdown();
  return exit_code;
}
