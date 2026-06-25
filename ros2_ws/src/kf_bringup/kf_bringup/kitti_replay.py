"""KITTI OXTS replay node — drives the EskfNode and checks parity with the Python reference.

Publish ordering (must match prototypes/python/eskf.py run_eskf step k):
    Python: predict(IMU[k-1]) -> update_gps(GPS[k] if present) -> record x_est[k].
    ROS: for step k, publish IMU[k-1] (stamped t[k]) FIRST, then GPS[k]. The node
    publishes the previous completed step on each IMU, so predict runs before update
    and the recorded states match the Python order exactly. A final duplicate-stamp
    IMU flushes the last step.
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from kf_msgs.msg import EgoState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Header


# ---------------------------------------------------------------------------
# Inline SO(3) helpers (mirror of prototypes/python/so3.py) — no import deps.
# ---------------------------------------------------------------------------

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler -> scalar-first quaternion [w, x, y, z]. Mirrors so3.euler_to_quat."""
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5);   sy = math.sin(yaw * 0.5)
    q = np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    ], dtype=float)
    norm = np.linalg.norm(q)
    q /= norm
    if q[0] < 0.0:
        q = -q          # canonical: positive real part
    return q


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion -> 3x3 rotation matrix. Mirrors so3.quat_to_rotmat."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=float)


def _stamp_from_secs(t: float):
    """Return a builtin_interfaces/Time from a float second value."""
    from builtin_interfaces.msg import Time
    msg = Time()
    msg.sec     = int(t)
    msg.nanosec = int(round((t - int(t)) * 1e9))
    return msg


# ---------------------------------------------------------------------------
# KittiReplay node
# ---------------------------------------------------------------------------

class KittiReplay(Node):
    def __init__(self) -> None:
        super().__init__("kitti_replay")

        # Parameters.
        self.declare_parameter("cache_path",
            "/workspace/data/cache/kitti_raw_2011_09_26_drive_0001_extract_oxts_v1.npz")
        self.declare_parameter("reference_path",
            "/workspace/data/cache/eskf_py_ref.npz")
        self.declare_parameter("gps_std_m",        0.75)
        self.declare_parameter("gps_rate_divisor", 10)
        self.declare_parameter("seed",             0)
        self.declare_parameter("publish_period_s", 0.001)

        cache_path   = self.get_parameter("cache_path").value
        ref_path     = self.get_parameter("reference_path").value
        gps_std_m    = self.get_parameter("gps_std_m").value
        gps_divisor  = self.get_parameter("gps_rate_divisor").value
        seed         = self.get_parameter("seed").value
        self._period = self.get_parameter("publish_period_s").value

        # Load cache.
        cache = np.load(cache_path)
        self._timestamps   = cache["timestamps"]          # (N,)
        self._enu          = cache["enu_position_m"]      # (N, 3)
        self._vel_body     = cache["velocity"]            # (N, 3) body frame
        self._rpy          = cache["roll_pitch_yaw"]      # (N, 3)
        self._accel_body   = cache["accel_body"]          # (N, 3)
        self._gyro_body    = cache["gyro_body"]           # (N, 3)

        N = self._timestamps.shape[0]

        # Build GPS measurements — same call order as build_gps_measurements() to get
        # identical RNG stream: indices first, then rng.normal with shape (len(indices),3).
        indices = np.arange(0, N, gps_divisor)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, gps_std_m, (len(indices), 3))
        gps_z = self._enu[indices] + noise
        self._gps_lookup: dict[int, np.ndarray] = {int(idx): gps_z[i] for i, idx in enumerate(indices)}

        # Build initial state: mirrors initial_state_from_oxts(sequence, 0).
        rpy0 = self._rpy[0]
        self._q0 = _euler_to_quat(rpy0[0], rpy0[1], rpy0[2])      # scalar-first ZYX
        R0 = _quat_to_rotmat(self._q0)
        self._pos0 = self._enu[0]                                   # ENU position
        self._vel0 = R0 @ self._vel_body[0]                        # body -> map velocity

        # Load Python reference trajectory.
        ref = np.load(ref_path)
        self._py_timestamps = ref["timestamps"]   # (N,)
        self._py_x_est      = ref["x_est"]        # (N, 3)

        # State for EgoState collection.
        self._received: list[np.ndarray] = []

        # QoS profiles.
        reliable_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2000,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        init_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Publishers.
        self._init_pub = self.create_publisher(Odometry,      "/eskf/init",  init_qos)
        self._imu_pub  = self.create_publisher(Imu,           "/imu/data",   reliable_qos)
        self._gps_pub  = self.create_publisher(PointStamped,  "/gps/fix",    reliable_qos)

        # Subscriber: collect ESKF output positions in order.
        self._ego_sub = self.create_subscription(
            EgoState, "/ego/state", self._on_ego, reliable_qos)

        self._n = N

    # ------------------------------------------------------------------
    # /eskf/init: publish the initial nominal state (transient_local).
    # ------------------------------------------------------------------
    def _publish_init(self) -> None:
        msg = Odometry()
        msg.header.stamp    = _stamp_from_secs(float(self._timestamps[0]))
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(self._pos0[0])
        msg.pose.pose.position.y = float(self._pos0[1])
        msg.pose.pose.position.z = float(self._pos0[2])
        msg.pose.pose.orientation.w = float(self._q0[0])   # scalar-first -> ROS w-first
        msg.pose.pose.orientation.x = float(self._q0[1])
        msg.pose.pose.orientation.y = float(self._q0[2])
        msg.pose.pose.orientation.z = float(self._q0[3])
        msg.twist.twist.linear.x = float(self._vel0[0])
        msg.twist.twist.linear.y = float(self._vel0[1])
        msg.twist.twist.linear.z = float(self._vel0[2])
        self._init_pub.publish(msg)
        self.get_logger().info(
            f"Published /eskf/init: pos=({self._pos0[0]:.2f},{self._pos0[1]:.2f},{self._pos0[2]:.2f})")

    # ------------------------------------------------------------------
    # Replay loop (run_eskf step ordering).
    # ------------------------------------------------------------------
    def run(self) -> None:
        # Publish init and allow the EskfNode subscriber to receive it before replay.
        self._publish_init()
        time.sleep(0.5)   # let transient_local deliver to late-joining node

        ts  = self._timestamps
        acc = self._accel_body
        gyr = self._gyro_body
        N   = self._n

        # Mirrors run_eskf: for k in 1..N-1:
        #   predict(IMU[k-1])  then  update_gps(GPS[k] if present)  then  record x_est[k].
        # Replay: for step k -> publish IMU[k-1] (stamped t[k]) FIRST, then GPS[k]. The node
        # publishes the previous completed step on each IMU, so predict happens before update.
        def publish_imu(idx_src: int, stamp_t: float) -> None:
            imu_msg = Imu()
            imu_msg.header.stamp    = _stamp_from_secs(stamp_t)   # t[k]: node dt = t[k]-t[k-1]
            imu_msg.header.frame_id = "base_link"
            imu_msg.linear_acceleration.x = float(acc[idx_src, 0])
            imu_msg.linear_acceleration.y = float(acc[idx_src, 1])
            imu_msg.linear_acceleration.z = float(acc[idx_src, 2])
            imu_msg.angular_velocity.x = float(gyr[idx_src, 0])
            imu_msg.angular_velocity.y = float(gyr[idx_src, 1])
            imu_msg.angular_velocity.z = float(gyr[idx_src, 2])
            self._imu_pub.publish(imu_msg)
            rclpy.spin_once(self, timeout_sec=self._period)

        for k in range(1, N):
            publish_imu(k - 1, float(ts[k]))   # IMU[k-1] -> predict step k

            if k in self._gps_lookup:           # GPS[k] -> update step k (after predict)
                z = self._gps_lookup[k]
                gps_msg = PointStamped()
                gps_msg.header.stamp    = _stamp_from_secs(float(ts[k]))
                gps_msg.header.frame_id = "map"
                gps_msg.point.x = float(z[0])
                gps_msg.point.y = float(z[1])
                gps_msg.point.z = float(z[2])
                self._gps_pub.publish(gps_msg)
                rclpy.spin_once(self, timeout_sec=self._period)

        # Flush: a duplicate-stamp IMU (dt==0) makes the node emit the final step without
        # predicting again, so all N-1 steps get published.
        publish_imu(N - 2, float(ts[N - 1]))

        self.get_logger().info(f"Replay done: published {N - 1} IMU steps + flush.")

        # Drain remaining /ego/state messages (up to 10 s timeout).
        self.get_logger().info("Waiting for remaining EgoState messages...")
        prev_count = -1
        deadline   = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self._received) == prev_count:
                break          # count stopped growing
            prev_count = len(self._received)

        self._check_parity()
        rclpy.shutdown()

    # ------------------------------------------------------------------
    # Collect EgoState positions.
    # ------------------------------------------------------------------
    def _on_ego(self, msg: EgoState) -> None:
        p = msg.pose.position
        self._received.append(np.array([p.x, p.y, p.z], dtype=float))

    # ------------------------------------------------------------------
    # Parity check against the Python reference.
    # ------------------------------------------------------------------
    def _check_parity(self) -> None:
        # The Python reference x_est[0] is the initial position (before any predict),
        # and x_est[k] is the position AFTER predict(IMU[k-1]) and optional GPS[k].
        # The node publishes on each IMU callback (steps 1..N-1), so received[i]
        # corresponds to Python x_est[i+1] (step index 1..N-1).
        # Skip index 0 (the init) and align by minimum available length.
        py_ref = self._py_x_est[1:]   # (N-1, 3) starting from step 1
        cpp    = np.array(self._received, dtype=float)  # (M, 3) collected from node

        n = min(len(py_ref), len(cpp))
        if n == 0:
            print("PARITY: FAIL  (no EgoState messages received)")
            return

        err_xy  = py_ref[:n, :2] - cpp[:n, :2]            # horizontal only, matches Python rmse
        rmse    = float(np.sqrt(np.mean(np.sum(err_xy * err_xy, axis=1))))
        max_abs = float(np.max(np.abs(py_ref[:n] - cpp[:n])))

        print(f"PARITY position_rmse_m={rmse:.4f} max_abs_m={max_abs:.4f} n={n}")
        if rmse < 0.1:
            print("PARITY: PASS")
        else:
            print("PARITY: FAIL")
            self.get_logger().warn(
                f"Parity FAIL: RMSE={rmse:.4f} m > 0.1 m threshold (n={n} steps compared)")


def main() -> None:
    rclpy.init()
    node = KittiReplay()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
