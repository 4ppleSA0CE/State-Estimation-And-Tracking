# prototypes/python/tracking/imm_filter.py
"""Per-track IMM: a bank of motion-model filters with Markov mode mixing. Config-driven —
`legacy` cfg (CV+CA+CT@trueω) reproduces imm_synthetic.py for the parity gate; `tracker`
cfg (CV+CA+CT±ω) + coast() is what the multi-target tracker runs. Generalized to n modes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from motion_models import build_model_bank

REF_DIM = 4
MEAS_DIM = 2
_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@dataclass
class IMMConfig:
    dt: float = 0.1
    sigma_pos: float = 1.0
    q_accel: float = 0.05
    omegas: tuple[float, ...] = (0.25, -0.25)   # CT turn rates; tracker default is ±ω
    pi_diag: float = 0.97
    mu0: tuple[float, ...] | None = None        # None → uniform over n modes
    p0_vel: float = 10.0

    @property
    def n_modes(self) -> int:
        return 2 + len(self.omegas)

    @property
    def pi_matrix(self) -> np.ndarray:
        n = self.n_modes
        off = (1.0 - self.pi_diag) / (n - 1)
        pi = np.full((n, n), off)
        np.fill_diagonal(pi, self.pi_diag)
        return pi

    @property
    def mu0_vec(self) -> np.ndarray:
        if self.mu0 is not None:
            return np.asarray(self.mu0, dtype=float)
        n = self.n_modes
        return np.full(n, 1.0 / n)


class IMMFilter:
    def __init__(self, cfg: IMMConfig, r: np.ndarray) -> None:
        self.cfg = cfg
        self.R = np.asarray(r, dtype=float)
        self.filters, self.mode_names = build_model_bank(
            cfg.dt, cfg.sigma_pos, cfg.q_accel, self.R, cfg.omegas
        )
        self.n = cfg.n_modes
        self.pi = cfg.pi_matrix
        self.mu = cfg.mu0_vec.copy()
        self._c: np.ndarray | None = None

    def init_state(self, x: np.ndarray, p: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float)
        for f in self.filters:
            f.set_state(x.copy(), p.copy())

    def _mix(self):
        states = [f.ref_state()[0] for f in self.filters]
        covs = [f.ref_state()[1] for f in self.filters]
        c = np.maximum(self.pi.T @ self.mu, 1e-12)
        mixed_x, mixed_p = [], []
        for j in range(self.n):
            x0 = np.zeros(REF_DIM)
            for i in range(self.n):
                x0 += (self.pi[i, j] * self.mu[i] / c[j]) * states[i]
            p0 = np.zeros((REF_DIM, REF_DIM))
            for i in range(self.n):
                dx = (states[i] - x0).reshape(REF_DIM, 1)
                p0 += (self.pi[i, j] * self.mu[i] / c[j]) * (covs[i] + dx @ dx.T)
            mixed_x.append(x0)
            mixed_p.append(p0)
        return mixed_x, mixed_p, c

    def predict(self) -> None:
        mixed_x, mixed_p, c = self._mix()
        self._c = c
        for j, f in enumerate(self.filters):
            f.set_state(mixed_x[j], mixed_p[j])
            f.predict()

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float).ravel()
        like = np.array([f.update(z) for f in self.filters])
        w = like * self._c
        denom = float(w.sum())
        if denom < 1e-300:
            self.mu = self.cfg.mu0_vec.copy()
        else:
            mu = np.maximum(w / denom, 1e-12)
            self.mu = mu / mu.sum()

    def coast(self) -> None:
        """Missed detection: predict() already advanced the modes predict-only; hold μ.
        The combined estimate from state() is therefore the predicted (coasted) one."""
        # intentionally empty — required change 1 (coast path). See docstring.

    def state(self):
        states = [f.ref_state()[0] for f in self.filters]
        covs = [f.ref_state()[1] for f in self.filters]
        x = np.zeros(REF_DIM)
        for j in range(self.n):
            x += self.mu[j] * states[j]
        p = np.zeros((REF_DIM, REF_DIM))
        for j in range(self.n):
            dx = (states[j] - x).reshape(REF_DIM, 1)
            p += self.mu[j] * (covs[j] + dx @ dx.T)
        return x, p

    def predicted_measurement(self):
        x, p = self.state()
        z_pred = _H @ x
        s = _H @ p @ _H.T + self.R
        return z_pred, s
