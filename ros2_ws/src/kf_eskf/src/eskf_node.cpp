// ROS 2 wrapper node for the 15-state error-state EKF.
//
// Publish/record ordering (matching prototypes/python/eskf.py run_eskf):
//   Python step k: predict(IMU[k-1]) -> optional update_gps(GPS[k]) -> record x_est[k].
//
// Replay ordering (kitti_replay.py): for step k it publishes IMU[k-1] (stamped t[k])
// THEN GPS[k] if present, so on a single-threaded executor predict fires before the
// update, exactly like Python. To record the POST-update state once per step, the node
// publishes the *previous* completed step at the start of each IMU callback (by then its
// predict and any GPS update are done), and the replay sends one final duplicate IMU to
// flush the last step. Result: one EgoState per step (1..N-1), post-update, in order.

#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <Eigen/Dense>

// ROS 2 core
#include <rclcpp/rclcpp.hpp>

// Message types
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

// TF2
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

// Project interfaces
#include <kf_msgs/msg/ego_state.hpp>
#include <kf_eskf/eskf.hpp>

namespace kf_eskf {

class EskfNode : public rclcpp::Node {
 public:
  EskfNode() : Node("eskf_node") {
    // Declare and read frame params.
    map_frame_  = declare_parameter<std::string>("map_frame",  "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");

    // Build EskfConfig from YAML params; defaults match EskfConfig struct.
    EskfConfig cfg;
    cfg.sigma_accel       = declare_parameter<double>("sigma_accel",       2.0);
    cfg.sigma_gyro        = declare_parameter<double>("sigma_gyro",        1e-3);
    cfg.sigma_accel_bias  = declare_parameter<double>("sigma_accel_bias",  1e-3);
    cfg.sigma_gyro_bias   = declare_parameter<double>("sigma_gyro_bias",   1e-5);
    cfg.gps_std_m         = declare_parameter<double>("gps_std_m",         0.75);

    // p0_att_deg: accept degrees in YAML, convert to radians for EskfConfig.
    const double p0_att_deg = declare_parameter<double>("p0_att_deg", 0.5);
    cfg.p0_pos        = declare_parameter<double>("p0_pos",        2.0);
    cfg.p0_vel        = declare_parameter<double>("p0_vel",        3.0);
    cfg.p0_att        = p0_att_deg * M_PI / 180.0;   // EskfConfig stores radians
    cfg.p0_accel_bias = declare_parameter<double>("p0_accel_bias", 0.1);
    cfg.p0_gyro_bias  = declare_parameter<double>("p0_gyro_bias",  1e-4);

    // lever_xyz: GPS antenna offset in body frame; default [0,0,0].
    auto lever_xyz = declare_parameter<std::vector<double>>("lever_xyz", {0.0, 0.0, 0.0});
    cfg.lever = Eigen::Vector3d(lever_xyz[0], lever_xyz[1], lever_xyz[2]);

    config_ = cfg;

    // Reliable, KeepLast(2000) QoS for all pub/subs so nothing drops during fast replay.
    auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(2000)).reliable();

    // /eskf/init: transient_local so late subscribers still receive the latched init.
    auto init_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                        .reliable()
                        .transient_local();

    // TF broadcaster.
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Publisher.
    ego_pub_ = create_publisher<kf_msgs::msg::EgoState>("/ego/state", reliable_qos);

    // Subscribers.
    init_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/eskf/init", init_qos,
        [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { onInit(msg); });

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", reliable_qos,
        [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { onImu(msg); });

    gps_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
        "/gps/fix", reliable_qos,
        [this](geometry_msgs::msg::PointStamped::ConstSharedPtr msg) { onGps(msg); });

    RCLCPP_INFO(get_logger(), "EskfNode ready (map_frame=%s, base_frame=%s)",
                map_frame_.c_str(), base_frame_.c_str());
  }

 private:
  // Initialize the nominal state once from /eskf/init.
  void onInit(nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    if (initialized_) return;  // only the first message counts

    const auto& pos = msg->pose.pose.position;
    const auto& ori = msg->pose.pose.orientation;
    const auto& lin = msg->twist.twist.linear;

    Eigen::Vector3d   position(pos.x, pos.y, pos.z);
    Eigen::Vector3d   velocity(lin.x, lin.y, lin.z);
    Eigen::Quaterniond q(ori.w, ori.x, ori.y, ori.z);   // Eigen ctor: (w,x,y,z)

    // Construct the filter; biases start at zero (KITTI OXTS is bias-corrected).
    eskf_ = std::make_unique<ErrorStateEkf>(position, velocity, q, config_);
    initialized_ = true;
    last_imu_stamp_ = rclcpp::Time(msg->header.stamp);  // t[0]; first dt is t[1]-t[0]
    RCLCPP_INFO(get_logger(), "ESKF initialized: pos=(%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
  }

  // GPS position update (ENU). Replay sends it AFTER this step's IMU, so it modifies the
  // pending step's state before that step is published at the next IMU.
  void onGps(geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {
    if (!initialized_) return;
    eskf_->updateGps({msg->point.x, msg->point.y, msg->point.z});
  }

  // IMU predict. Each IMU means the PREVIOUS step is fully complete (its predict + any GPS
  // update are done), so publish that first, then predict this step.
  void onImu(sensor_msgs::msg::Imu::ConstSharedPtr msg) {
    if (!initialized_) return;

    if (have_pending_) publishState(pending_stamp_);  // emit the now-complete previous step

    const rclcpp::Time stamp(msg->header.stamp);
    const double dt = (stamp - last_imu_stamp_).seconds();
    last_imu_stamp_ = stamp;
    if (dt <= 0.0) return;  // duplicate flush IMU (dt==0): published pending, nothing to predict

    eskf_->predict(
        {msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z},
        {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z},
        dt);
    have_pending_   = true;     // this step becomes publishable once the next IMU (or flush) arrives
    pending_stamp_  = stamp;
  }

  // Publish EgoState and broadcast map->base_link TF.
  void publishState(const rclcpp::Time& stamp) {
    const Eigen::Vector3d&    p = eskf_->position();
    const Eigen::Vector3d&    v = eskf_->velocity();
    const Eigen::Quaterniond& q = eskf_->quaternion();
    const Eigen::Vector3d& ba   = eskf_->accelBias();
    const Eigen::Vector3d& bg   = eskf_->gyroBias();

    // --- EgoState ---
    kf_msgs::msg::EgoState ego;
    ego.header.stamp    = stamp;
    ego.header.frame_id = map_frame_;

    ego.pose.position.x    = p.x();
    ego.pose.position.y    = p.y();
    ego.pose.position.z    = p.z();
    ego.pose.orientation.w = q.w();
    ego.pose.orientation.x = q.x();
    ego.pose.orientation.y = q.y();
    ego.pose.orientation.z = q.z();

    ego.twist.linear.x = v.x();
    ego.twist.linear.y = v.y();
    ego.twist.linear.z = v.z();

    ego.accel_bias.x = ba.x();
    ego.accel_bias.y = ba.y();
    ego.accel_bias.z = ba.z();

    ego.gyro_bias.x = bg.x();
    ego.gyro_bias.y = bg.y();
    ego.gyro_bias.z = bg.z();

    // 15x15 covariance, row-major into float64[225].
    const Matrix15d& P = eskf_->covariance();
    for (int r = 0; r < 15; ++r)
      for (int c = 0; c < 15; ++c)
        ego.covariance[static_cast<size_t>(r * 15 + c)] = P(r, c);

    ego.filter_type = kf_msgs::msg::EgoState::FILTER_ESKF;
    ego_pub_->publish(ego);

    // --- TF: map -> base_link ---
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp            = stamp;
    tf.header.frame_id         = map_frame_;
    tf.child_frame_id          = base_frame_;
    tf.transform.translation.x = p.x();
    tf.transform.translation.y = p.y();
    tf.transform.translation.z = p.z();
    tf.transform.rotation.w    = q.w();
    tf.transform.rotation.x    = q.x();
    tf.transform.rotation.y    = q.y();
    tf.transform.rotation.z    = q.z();
    tf_broadcaster_->sendTransform(tf);
  }

  // Config & state.
  EskfConfig config_;
  std::unique_ptr<ErrorStateEkf> eskf_;
  bool initialized_   = false;
  bool have_pending_  = false;                          // a predicted step awaiting publish
  rclcpp::Time last_imu_stamp_{0, 0, RCL_ROS_TIME};     // previous IMU stamp, for dt
  rclcpp::Time pending_stamp_{0, 0, RCL_ROS_TIME};      // stamp of the pending step

  // Frames.
  std::string map_frame_;
  std::string base_frame_;

  // ROS handles.
  rclcpp::Publisher<kf_msgs::msg::EgoState>::SharedPtr ego_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     init_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr       imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr gps_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace kf_eskf

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  // Single-threaded executor: callbacks are serialized so GPS fires before IMU when
  // the replay publishes them in that order for each step.
  rclcpp::spin(std::make_shared<kf_eskf::EskfNode>());
  rclcpp::shutdown();
  return 0;
}
