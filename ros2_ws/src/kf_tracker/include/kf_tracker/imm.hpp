// Per-track IMM: a bank of motion-model filters with Markov mode mixing.
//
// A 1:1 C++ port of prototypes/python/tracking/imm_filter.py plus motion_models.build_model_bank.
// Bank order is FIXED as CV, CA, then one CT per entry of `omegas`, so mode indices line up with
// the Python reference used by the parity gate. All I/O is in models.hpp's 4-D reference space.
#ifndef KF_TRACKER_IMM_HPP
#define KF_TRACKER_IMM_HPP

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "kf_tracker/models.hpp"

namespace kf_tracker {

struct ImmConfig {
  double dt = 0.1;
  // Mirrors IMMConfig.sigma_pos so configs port 1:1, but the filters take R from ImmFilter's `r`
  // argument alone. Callers that want R = sigma_pos^2 * I must pass it explicitly.
  double sigma_pos = 1.0;
  double q_accel = 0.05;
  std::vector<double> omegas{0.25, -0.25};   // CT turn rates; the tracker default is +/- omega
  double pi_diag = 0.97;
  std::vector<double> mu0;                   // empty = uniform over n modes
  // Explicit transition matrix, ROW-MAJOR n*n; empty = build one from pi_diag. The generated
  // matrix is always SYMMETRIC, so with pi_diag alone a pi/pi^T swap anywhere in the mixing
  // layer is unobservable. This is the injection point tests use to supply an asymmetric pi;
  // Python gets the same hook for free by assigning IMMFilter.pi after construction.
  std::vector<double> pi_override;

  int numModes() const { return 2 + static_cast<int>(omegas.size()); }

  Eigen::MatrixXd piMatrix() const {
    const int n = numModes();
    if (!pi_override.empty()) {
      if (pi_override.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(n))
        throw std::invalid_argument("ImmConfig::pi_override must hold exactly numModes()^2 entries");
      using RowMajorXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      return Eigen::MatrixXd(Eigen::Map<const RowMajorXd>(pi_override.data(), n, n));
    }
    const double off = (1.0 - pi_diag) / static_cast<double>(n - 1);
    Eigen::MatrixXd pi = Eigen::MatrixXd::Constant(n, n, off);
    pi.diagonal().setConstant(pi_diag);
    return pi;
  }

  Eigen::VectorXd mu0Vec() const {
    const int n = numModes();
    if (!mu0.empty())
      return Eigen::Map<const Eigen::VectorXd>(mu0.data(), static_cast<int>(mu0.size()));
    return Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n));
  }
};

class ImmFilter {
 public:
  ImmFilter(const ImmConfig& cfg, const Matrix2d& r) : cfg_(cfg), r_(r) {
    pi_ = cfg.piMatrix();
    mu_ = cfg.mu0Vec();
    // Python leaves _c = None until the first predict(); seeding it with mu0 keeps an
    // update()-before-predict() call defined instead of undefined. No parity path hits it.
    c_ = mu_;

    // `r` is the only measurement-noise input the linear modes take; cfg.sigma_pos is the
    // caller's handle for building it (see ImmConfig::sigma_pos) and deliberately does not
    // reach them — see the MEASUREMENT NOISE note in models.hpp.
    filters_.emplace_back(std::make_unique<CvModelFilter>(cfg.dt, cfg.q_accel, r));
    names_.emplace_back("CV");
    filters_.emplace_back(std::make_unique<CaModelFilter>(cfg.dt, cfg.q_accel, r));
    names_.emplace_back("CA");
    for (double w : cfg.omegas) {
      filters_.emplace_back(std::make_unique<CtModelFilter>(cfg.dt, w, cfg.q_accel, r));
      // Default double formatting is %g-equivalent, so this matches Python's
      // f"CT{'+' if w >= 0 else ''}{w:g}" — 0.25 -> "CT+0.25", -0.25 -> "CT-0.25".
      std::ostringstream os;
      os << "CT" << (w >= 0.0 ? "+" : "") << w;
      names_.emplace_back(os.str());
    }
  }

  int numModes() const { return static_cast<int>(filters_.size()); }
  const std::vector<std::string>& modeNames() const { return names_; }
  const Eigen::MatrixXd& transitionMatrix() const { return pi_; }
  const Eigen::VectorXd& modeProbabilities() const { return mu_; }

  // Seed every mode with its OWN copy of the state (setState takes const refs and copies by
  // value, so no aliasing — Python needed an explicit .copy() per mode to get the same effect).
  void initState(const Vector4d& x, const Matrix4d& p) {
    for (auto& f : filters_) f->setState(x, p);
  }

  void predict() {
    std::vector<Vector4d> mixed_x;
    std::vector<Matrix4d> mixed_p;
    mix(mixed_x, mixed_p);
    for (std::size_t j = 0; j < filters_.size(); ++j) {
      filters_[j]->setState(mixed_x[j], mixed_p[j]);
      filters_[j]->predict();
    }
  }

  void update(const Vector2d& z) {
    Eigen::VectorXd like(numModes());
    for (std::size_t j = 0; j < filters_.size(); ++j)
      like(static_cast<int>(j)) = filters_[j]->update(z);

    const Eigen::VectorXd w = like.cwiseProduct(c_);
    const double denom = w.sum();
    if (denom < 1e-300) {   // total-likelihood underflow: fall back to the prior
      mu_ = cfg_.mu0Vec();
      return;
    }
    const Eigen::VectorXd mu = (w / denom).cwiseMax(1e-12);   // floor, then renormalize to 1
    mu_ = mu / mu.sum();
  }

  // Missed detection: predict() has already advanced every mode predict-only, and mu is
  // deliberately HELD, so state() returns the coasted estimate. Intentionally does nothing —
  // do not "fix" this by adding a mode update.
  void coast() {}

  void state(Vector4d& x, Matrix4d& p) const {
    const std::size_t n = filters_.size();
    std::vector<Vector4d> xs(n);
    std::vector<Matrix4d> ps(n);
    for (std::size_t j = 0; j < n; ++j) filters_[j]->refState(xs[j], ps[j]);

    x.setZero();
    for (std::size_t j = 0; j < n; ++j) x += mu_(static_cast<int>(j)) * xs[j];

    p.setZero();
    for (std::size_t j = 0; j < n; ++j) {
      const Vector4d dx = xs[j] - x;
      p += mu_(static_cast<int>(j)) * (ps[j] + dx * dx.transpose());
    }
  }

  void predictedMeasurement(Vector2d& z_pred, Matrix2d& s) const {
    Vector4d x;
    Matrix4d p;
    state(x, p);
    z_pred = x.head<kMeasDim>();
    s = p.topLeftCorner<kMeasDim, kMeasDim>() + r_;
  }

 private:
  // Markov mixing: c(j) = sum_i pi(i,j) mu(i), then mode j is re-seeded with the c(j)-normalized
  // blend of every mode. c is floored at 1e-12 so a collapsed mode cannot divide by zero.
  void mix(std::vector<Vector4d>& mixed_x, std::vector<Matrix4d>& mixed_p) {
    const std::size_t n = filters_.size();
    std::vector<Vector4d> xs(n);
    std::vector<Matrix4d> ps(n);
    for (std::size_t j = 0; j < n; ++j) filters_[j]->refState(xs[j], ps[j]);

    c_ = (pi_.transpose() * mu_).cwiseMax(1e-12);

    mixed_x.assign(n, Vector4d::Zero());
    mixed_p.assign(n, Matrix4d::Zero());
    for (int j = 0; j < static_cast<int>(n); ++j) {
      Vector4d& x0 = mixed_x[static_cast<std::size_t>(j)];
      for (int i = 0; i < static_cast<int>(n); ++i)
        x0 += (pi_(i, j) * mu_(i) / c_(j)) * xs[static_cast<std::size_t>(i)];

      Matrix4d& p0 = mixed_p[static_cast<std::size_t>(j)];
      for (int i = 0; i < static_cast<int>(n); ++i) {
        const Vector4d dx = xs[static_cast<std::size_t>(i)] - x0;
        p0 += (pi_(i, j) * mu_(i) / c_(j)) * (ps[static_cast<std::size_t>(i)] + dx * dx.transpose());
      }
    }
  }

  ImmConfig cfg_;
  Matrix2d r_;
  std::vector<std::unique_ptr<ModelFilter>> filters_;
  std::vector<std::string> names_;
  Eigen::MatrixXd pi_;
  Eigen::VectorXd mu_;
  Eigen::VectorXd c_;
};

}  // namespace kf_tracker

#endif  // KF_TRACKER_IMM_HPP
