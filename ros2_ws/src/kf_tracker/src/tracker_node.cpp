// ROS 2 wrapper node for the IMM multi-target tracker.
//
// One DetectionArray in -> one full tracker step -> one TrackArray out, published from the same
// callback. Unlike eskf_node there is no pending-state buffering: the ESKF needed that only
// because IMU and GPS arrived on separate topics, whereas a detection frame is atomic.
//
// Tracking happens in the frame the detections arrive in -- no ego transform, no /ego/state
// subscription. Transforming detections base_link->map is Stage 6's job (design doc decision D2),
// so the published TrackArray simply echoes the incoming header.
//
// Exception policy: rclcpp does NOT catch exceptions thrown out of a subscription callback -- they
// unwind through spin() and reach std::terminate. The kernel throws by design in two places that
// sit on this callback's hot path (kf_common::sigmaPoints -> std::runtime_error on a covariance
// that is still non-PSD after its jitter retry, reached via the CT mode's predict() for every
// track every frame; kf_common::hungarian -> std::invalid_argument on a non-finite cost matrix).
// One degenerate track must not kill a multi-target node, so the whole step is wrapped and a
// failure degrades to an empty frame -- see onDetections().
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

// ROS 2 core
#include <rclcpp/rclcpp.hpp>

// Message types
#include <geometry_msgs/msg/quaternion.hpp>
#include <kf_msgs/msg/detection.hpp>
#include <kf_msgs/msg/detection_array.hpp>
#include <kf_msgs/msg/track.hpp>
#include <kf_msgs/msg/track_array.hpp>
#include <std_msgs/msg/header.hpp>

// Project interfaces
#include <kf_tracker/tracker.hpp>

namespace kf_tracker {
namespace {

// Yaw about the vertical axis from a quaternion, matching the Box3D yaw convention (KITTI
// rotation_y). The replay node encodes yaw the same way; a convention mismatch here would not
// throw, it would silently wreck IoU association, so the two must stay in lockstep.
double yawFromQuaternion(const geometry_msgs::msg::Quaternion& q) {
  return std::atan2(2.0 * (q.w * q.y + q.x * q.z), 1.0 - 2.0 * (q.y * q.y + q.x * q.x));
}

std::string joinDoubles(const std::vector<double>& values) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) oss << ", ";
    oss << values[i];
  }
  return oss.str();
}

}  // namespace

class TrackerNode : public rclcpp::Node {
 public:
  TrackerNode() : Node("tracker_node") {
    // Build TrackerConfig from YAML params; names and defaults mirror
    // kf_bringup/config/tracker.yaml. Integers must be declared as int64_t (the only integral
    // parameter type ROS 2 has) and narrowed.
    TrackerConfig cfg;
    cfg.imm.dt        = declare_parameter<double>("dt",        0.1);
    cfg.imm.sigma_pos = declare_parameter<double>("sigma_pos", 0.5);
    cfg.imm.q_accel   = declare_parameter<double>("q_accel",   2.0);
    cfg.imm.omegas    = declare_parameter<std::vector<double>>("omegas", {0.2, -0.2});
    cfg.imm.pi_diag   = declare_parameter<double>("pi_diag",   0.97);
    cfg.cost          = declare_parameter<std::string>("cost", "iou");
    cfg.iou_gate      = declare_parameter<double>("iou_gate",  0.01);
    cfg.min_hits      = static_cast<int>(declare_parameter<int64_t>("min_hits", 3));
    cfg.max_age       = static_cast<int>(declare_parameter<int64_t>("max_age",  2));
    cfg.greedy        = declare_parameter<bool>("greedy",      false);
    cfg.p0_vel        = declare_parameter<double>("p0_vel",    10.0);

    // KittiTracker::costMatrix tests `cost == "iou"` and falls through to the MAHALANOBIS branch
    // for anything else, so a typo ("IOU", "iou3d") silently swaps the association metric instead
    // of failing. Reject it here; main() catches and logs FATAL.
    if (cfg.cost != "iou" && cfg.cost != "maha")
      throw std::invalid_argument("parameter `cost` must be \"iou\" or \"maha\", got \"" +
                                  cfg.cost + "\"");

    // Bank order is CV, CA, then one CT per entry of omegas.
    const std::size_t num_modes = cfg.imm.omegas.size() + 2;

    tracker_ = std::make_unique<KittiTracker>(cfg);

    // Reliable, KeepLast(2000) QoS so nothing drops during fast replay -- same as eskf_node.
    auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(2000)).reliable();

    track_pub_ = create_publisher<kf_msgs::msg::TrackArray>("/tracks", reliable_qos);

    det_sub_ = create_subscription<kf_msgs::msg::DetectionArray>(
        "/detections", reliable_qos,
        [this](kf_msgs::msg::DetectionArray::ConstSharedPtr msg) { onDetections(msg); });

    // Log the fully resolved config once, mode count included, so a mis-set `omegas` is visible
    // at startup instead of showing up as a silent parity failure.
    RCLCPP_INFO(get_logger(),
                "TrackerNode ready (cost=%s, min_hits=%d, max_age=%d, modes=%zu, dt=%.4g, "
                "sigma_pos=%.4g, q_accel=%.4g, pi_diag=%.4g, iou_gate=%.4g, greedy=%s, "
                "p0_vel=%.4g, omegas=[%s])",
                cfg.cost.c_str(), cfg.min_hits, cfg.max_age, num_modes, cfg.imm.dt,
                cfg.imm.sigma_pos, cfg.imm.q_accel, cfg.imm.pi_diag, cfg.iou_gate,
                cfg.greedy ? "true" : "false", cfg.p0_vel, joinDoubles(cfg.imm.omegas).c_str());
  }

 private:
  // One detection frame -> one tracker step -> one TrackArray, all in this callback.
  void onDetections(kf_msgs::msg::DetectionArray::ConstSharedPtr msg) {
    kf_msgs::msg::TrackArray out;
    out.header = msg->header;  // echo the detection stamp + frame (decision D2: no ego transform)

    try {
      std::vector<Box3D> dets;
      dets.reserve(msg->detections.size());
      for (const auto& d : msg->detections) {
        Box3D b;
        b.x   = d.pose.position.x;
        b.y   = d.pose.position.y;
        b.z   = d.pose.position.z;
        b.yaw = yawFromQuaternion(d.pose.orientation);
        b.l   = d.dimensions.x;   // Detection.dimensions is (length, width, height) in that order
        b.w   = d.dimensions.y;
        b.h   = d.dimensions.z;
        b.track_id = d.object_id;
        // Detection.msg carries no score field, so Box3D's default (1.0) stands.
        dets.push_back(b);
      }

      // An EMPTY frame must still step the tracker so every track coasts and ages -- do not
      // early-return on msg->detections.empty().
      const std::vector<const BoxTrack*> confirmed = tracker_->step(dets);

      // Snapshot each track NOW. These pointers are invalidated by the next step(), and
      // modeProbabilities() hands back a const& into live, mutating filter state, so everything
      // must be copied into the message inside this loop and nothing may outlive the callback.
      out.tracks.reserve(confirmed.size());
      for (const BoxTrack* t : confirmed) {
        const Box3D box = t->box();
        Vector4d x;
        Matrix4d p;
        t->immState(x, p);

        kf_msgs::msg::Track track;
        track.id = t->id;

        track.pose.position.x = box.x;
        track.pose.position.y = box.y;
        track.pose.position.z = box.z;
        // Pure yaw about the vertical axis: x = z = 0, the exact inverse of yawFromQuaternion.
        track.pose.orientation.w = std::cos(box.yaw * 0.5);
        track.pose.orientation.x = 0.0;
        track.pose.orientation.y = std::sin(box.yaw * 0.5);
        track.pose.orientation.z = 0.0;

        track.dimensions.x = box.l;
        track.dimensions.y = box.w;
        track.dimensions.z = box.h;

        // BEV motion state [x, z, vx, vz] and its 4x4 covariance, row-major.
        track.state = {x(0), x(1), x(2), x(3)};
        track.covariance.resize(16);
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            track.covariance[static_cast<std::size_t>(r * 4 + c)] = p(r, c);
          }
        }

        // mode_probabilities is unbounded: its length is the bank size, not a fixed 3.
        const Eigen::VectorXd& mu = t->modeProbabilities();
        track.mode_probabilities.assign(mu.data(), mu.data() + mu.size());

        track.age          = t->age;
        track.missed_count = t->time_since_update;

        out.tracks.push_back(track);
      }
    } catch (const std::exception& e) {
      onStepFailure(msg->header, out, e.what());
    } catch (...) {
      onStepFailure(msg->header, out, "unknown exception");
    }

    // Published unconditionally: /tracks stays 1:1 with /detections even on a failed frame, so a
    // downstream consumer indexing by frame does not silently shift. Guarded too -- publish()
    // reaches rcl and throws rclcpp::exceptions::RCLError on a middleware failure, which would
    // otherwise be the one path left that escapes this callback into std::terminate.
    try {
      track_pub_->publish(out);
    } catch (const std::exception& e) {
      ++publish_failures_;
      RCLCPP_ERROR_THROTTLE(get_logger(), throttle_clock_, 1000,
                            "publishing /tracks failed at stamp %d.%09u: %s (total failures=%llu)",
                            msg->header.stamp.sec, msg->header.stamp.nanosec, e.what(),
                            static_cast<unsigned long long>(publish_failures_));
    }
  }

  // Failure policy: publish an EMPTY TrackArray for this frame rather than a partially filled one.
  // When step() throws we never receive the track pointers at all, so there is nothing partial to
  // publish; and if the fill loop itself were to throw, a half-populated frame would read as
  // "those tracks died" -- indistinguishable from a legitimate result and strictly more misleading
  // than an explicit empty frame paired with an error log. The node stays alive: losing one frame
  // beats terminating a multi-target tracker over one degenerate covariance.
  void onStepFailure(const std_msgs::msg::Header& header, kf_msgs::msg::TrackArray& out,
                     const char* what) {
    out.tracks.clear();
    ++step_failures_;
    RCLCPP_ERROR_THROTTLE(get_logger(), throttle_clock_, 1000,
                          "tracker step failed at stamp %d.%09u: %s; publishing an empty "
                          "TrackArray for this frame (total failures=%llu)",
                          header.stamp.sec, header.stamp.nanosec, what,
                          static_cast<unsigned long long>(step_failures_));
  }

  std::unique_ptr<KittiTracker> tracker_;
  std::uint64_t step_failures_ = 0;
  std::uint64_t publish_failures_ = 0;
  // Steady clock for log throttling: a ROS-time clock would stall the throttle window under
  // use_sim_time and suppress every error after the first.
  rclcpp::Clock throttle_clock_{RCL_STEADY_TIME};

  // ROS handles.
  rclcpp::Publisher<kf_msgs::msg::TrackArray>::SharedPtr track_pub_;
  rclcpp::Subscription<kf_msgs::msg::DetectionArray>::SharedPtr det_sub_;
};

}  // namespace kf_tracker

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  int exit_code = 0;
  try {
    // Single-threaded executor: one detection frame is fully processed before the next starts.
    rclcpp::spin(std::make_shared<kf_tracker::TrackerNode>());
  } catch (const std::exception& e) {
    // Construction-time failures (e.g. an unrecognised `cost` string) would otherwise reach
    // std::terminate with no diagnostic.
    RCLCPP_FATAL(rclcpp::get_logger("tracker_node"), "tracker_node aborting: %s", e.what());
    exit_code = 1;
  }
  rclcpp::shutdown();
  return exit_code;
}
