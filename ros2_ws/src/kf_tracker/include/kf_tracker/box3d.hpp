// 3D box in KITTI camera coordinates + oriented 3D-IoU.
//
// A 1:1 C++ port of prototypes/python/tracking/kitti_boxes.py. BEV is the ground plane (x, z);
// y points DOWN, so a box occupies [y - h, y] vertically. IoU = BEV-polygon overlap x height
// overlap / union volume. ROS-independent: standard library only.
#ifndef KF_TRACKER_BOX3D_HPP
#define KF_TRACKER_BOX3D_HPP

#include <algorithm>   // std::reverse, std::max, std::min
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>     // std::pair
#include <vector>

namespace kf_tracker {

struct Box3D {
  double x = 0.0;
  double y = 0.0;        // bottom face (y is down in KITTI camera coords)
  double z = 0.0;
  double yaw = 0.0;      // KITTI rotation_y, rad
  double l = 0.0;
  double w = 0.0;
  double h = 0.0;
  double score = 1.0;
  int track_id = -1;

  std::pair<double, double> centerBev() const { return {x, z}; }
};

namespace detail {

using Point = std::pair<double, double>;
using Polygon = std::vector<Point>;

// Four BEV corners. Mirrors _bev_corners: local (l/2, w/2) offsets rotated by yaw with the
// KITTI sign convention (x' = c*xl + s*zl, z' = -s*xl + c*zl).
inline Polygon bevCorners(const Box3D& b) {
  const double c = std::cos(b.yaw);
  const double s = std::sin(b.yaw);
  const double hl = b.l / 2.0;
  const double hw = b.w / 2.0;
  const std::array<Point, 4> local{{{hl, hw}, {hl, -hw}, {-hl, -hw}, {-hl, hw}}};

  Polygon out;
  out.reserve(4);
  for (const auto& p : local)
    out.emplace_back(c * p.first + s * p.second + b.x, -s * p.first + c * p.second + b.z);
  return out;
}

inline double signedArea2(const Polygon& poly) {
  double s = 0.0;
  const size_t n = poly.size();
  for (size_t i = 0; i < n; ++i) {
    const Point& a = poly[i];
    const Point& b = poly[(i + 1) % n];
    s += a.first * b.second - b.first * a.second;
  }
  return s;
}

inline Polygon ensureCcw(Polygon poly) {
  if (signedArea2(poly) < 0.0) std::reverse(poly.begin(), poly.end());
  return poly;
}

inline double polyArea(const Polygon& poly) {
  if (poly.size() < 3) return 0.0;
  return std::abs(signedArea2(poly)) * 0.5;
}

inline bool inside(const Point& p, const Point& a, const Point& b) {
  return (b.first - a.first) * (p.second - a.second) -
             (b.second - a.second) * (p.first - a.first) >=
         0.0;
}

inline Point intersect(const Point& p1, const Point& p2, const Point& a, const Point& b) {
  const double x1 = p1.first, z1 = p1.second;
  const double x2 = p2.first, z2 = p2.second;
  const double x3 = a.first, z3 = a.second;
  const double x4 = b.first, z4 = b.second;
  const double den = (x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4);
  if (std::abs(den) < 1e-12) return p2;   // parallel: degenerate, mirrors the Python fallback
  const double t = ((x1 - x3) * (z3 - z4) - (z1 - z3) * (x3 - x4)) / den;
  return {x1 + t * (x2 - x1), z1 + t * (z2 - z1)};
}

// Sutherland-Hodgman: clip `subject` by the convex, CCW polygon `clip`.
inline Polygon clipPolygon(const Polygon& subject, const Polygon& clip) {
  Polygon output = subject;
  const size_t n = clip.size();
  for (size_t i = 0; i < n; ++i) {
    if (output.empty()) break;
    const Point& a = clip[i];
    const Point& b = clip[(i + 1) % n];
    const Polygon input = output;
    output.clear();
    const size_t k = input.size();
    for (size_t j = 0; j < k; ++j) {
      const Point& cur = input[j];
      const Point& prv = input[(j + k - 1) % k];   // Python's input[j-1] wraps to the last element
      if (inside(cur, a, b)) {
        if (!inside(prv, a, b)) output.push_back(intersect(prv, cur, a, b));
        output.push_back(cur);
      } else if (inside(prv, a, b)) {
        output.push_back(intersect(prv, cur, a, b));
      }
    }
  }
  return output;
}

inline double bevIntersectionArea(const Box3D& a, const Box3D& b) {
  return polyArea(clipPolygon(ensureCcw(bevCorners(a)), ensureCcw(bevCorners(b))));
}

}  // namespace detail

inline double iouBev(const Box3D& a, const Box3D& b) {
  const double inter = detail::bevIntersectionArea(a, b);
  const double uni = a.l * a.w + b.l * b.w - inter;
  return uni > 0.0 ? inter / uni : 0.0;
}

inline double iou3d(const Box3D& a, const Box3D& b) {
  const double inter_area = detail::bevIntersectionArea(a, b);
  const double lo = std::max(a.y - a.h, b.y - b.h);
  const double hi = std::min(a.y, b.y);
  const double h_overlap = std::max(0.0, hi - lo);
  const double inter_vol = inter_area * h_overlap;
  const double uni = a.l * a.w * a.h + b.l * b.w * b.h - inter_vol;
  return uni > 0.0 ? inter_vol / uni : 0.0;
}

}  // namespace kf_tracker

#endif  // KF_TRACKER_BOX3D_HPP
