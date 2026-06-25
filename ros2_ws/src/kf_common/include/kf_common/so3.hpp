// Scalar-first SO(3) / quaternion helpers, a C++ mirror of prototypes/python/so3.py.
//
// Convention (must match the Python prototype so the C++ node can be validated
// against it to 0.1 m RMSE): quaternions are unit, scalar-first [w, x, y, z], and
// rotate body vectors into the world/map frame. The error-state filters use the
// GLOBAL angular-error convention q_true = expMap(dtheta) * q_nominal.
#ifndef KF_COMMON_SO3_HPP
#define KF_COMMON_SO3_HPP

#include <Eigen/Dense>
#include <cmath>

namespace kf_common {

// Smallest angle below which we switch to the small-angle series to avoid 0/0.
inline constexpr double kSmallAngle = 1e-12;

// Skew-symmetric matrix: skew(v) * w == v.cross(w). Shows up in every rotation
// Jacobian because d/dtheta (R(theta) a) at theta=0 is -skew(a).
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d s;
  s <<    0.0, -v.z(),  v.y(),
       v.z(),    0.0, -v.x(),
      -v.y(),  v.x(),    0.0;
  return s;
}

// Normalize to a unit scalar-first quaternion with a non-negative real part
// (so q and -q, which represent the same rotation, have one canonical form).
inline Eigen::Quaterniond normalize(const Eigen::Quaterniond& q) {
  Eigen::Quaterniond n = q.normalized();
  if (n.w() < 0.0) {
    n.coeffs() *= -1.0;  // Eigen stores coeffs as (x, y, z, w); flip the whole sign
  }
  return n;
}

// Exp map (rotation vector -> quaternion), the so3.py rotvec_to_quat.
inline Eigen::Quaterniond expMap(const Eigen::Vector3d& rotvec) {
  const double angle = rotvec.norm();
  if (angle < kSmallAngle) {
    // small angle: sin(a/2) ~ a/2, cos(a/2) ~ 1, so q ~ [1, 0.5*rotvec]
    return normalize(Eigen::Quaterniond(1.0, 0.5 * rotvec.x(), 0.5 * rotvec.y(), 0.5 * rotvec.z()));
  }
  const Eigen::Vector3d axis = rotvec / angle;          // unit rotation axis
  const double half = 0.5 * angle;                      // half-angle for the quaternion
  const double s = std::sin(half);
  return normalize(Eigen::Quaterniond(std::cos(half), axis.x() * s, axis.y() * s, axis.z() * s));
}

// Log map (quaternion -> rotation vector), the so3.py quat_to_rotvec.
inline Eigen::Vector3d logMap(const Eigen::Quaterniond& q) {
  const Eigen::Quaterniond n = normalize(q);
  const Eigen::Vector3d vec(n.x(), n.y(), n.z());       // imaginary part = sin(angle/2) * axis
  const double sin_half = vec.norm();
  if (sin_half < kSmallAngle) {
    return 2.0 * vec;                                   // near identity: angle ~ 2 * imaginary part
  }
  const double angle = 2.0 * std::atan2(sin_half, n.w());
  return (angle / sin_half) * vec;
}

// Active rotation matrix (DCM) from a scalar-first quaternion.
inline Eigen::Matrix3d toRotationMatrix(const Eigen::Quaterniond& q) {
  return normalize(q).toRotationMatrix();
}

// Global (world-frame) boxplus on attitude: q_new = expMap(dtheta) * q. Left-multiply
// because the error rotation lives in the map frame, matching the ESKF convention.
inline Eigen::Quaterniond boxplus(const Eigen::Quaterniond& q, const Eigen::Vector3d& dtheta) {
  return normalize(expMap(dtheta) * q);
}

// Global boxminus: the rotation-vector error dtheta such that a == boxplus(b, dtheta).
inline Eigen::Vector3d boxminus(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  return logMap(normalize(a) * normalize(b).conjugate());
}

}  // namespace kf_common

#endif  // KF_COMMON_SO3_HPP
