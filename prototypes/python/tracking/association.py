# prototypes/python/tracking/association.py
"""GNN data association: χ²-gated Mahalanobis cost, solved by Hungarian (scipy) with an
AB3DMOT-style greedy fallback. Cost uses each track's combined innovation covariance S."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

GATE_DIM = 2
DEFAULT_GATE = float(chi2.ppf(0.99, GATE_DIM))     # ≈ 9.21 for 2-D position
BIG_COST = 1e6                                       # gated-out pairs


def mahalanobis_sq(y: np.ndarray, s: np.ndarray) -> float:
    return float(y @ np.linalg.solve(s, y))


def build_cost_matrix(preds, dets, gate=DEFAULT_GATE) -> np.ndarray:
    """preds: list of (z_pred, S); dets: (M,2). Gated pairs get BIG_COST."""
    n, m = len(preds), len(dets)
    cost = np.full((n, m), BIG_COST)
    for i, (z_pred, s) in enumerate(preds):
        for j in range(m):
            d2 = mahalanobis_sq(dets[j] - z_pred, s)
            if d2 <= gate:
                cost[i, j] = d2
    return cost


def _greedy(cost, gate):
    n, m = cost.shape
    pairs = sorted(
        ((cost[i, j], i, j) for i in range(n) for j in range(m) if cost[i, j] <= gate),
        key=lambda t: t[0],
    )
    used_t, used_d, matches = set(), set(), []
    for _, i, j in pairs:
        if i not in used_t and j not in used_d:
            matches.append((i, j))
            used_t.add(i)
            used_d.add(j)
    return matches


def associate(preds, dets, gate=DEFAULT_GATE, greedy=False):
    """Return (matches, unmatched_dets, unmatched_tracks). matches is a list of (track_i, det_j)."""
    dets = np.asarray(dets, dtype=float).reshape(-1, 2)
    n, m = len(preds), len(dets)
    if n == 0 or m == 0:
        return [], list(range(m)), list(range(n))
    cost = build_cost_matrix(preds, dets, gate)
    if greedy:
        matches = _greedy(cost, gate)
    else:
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] <= gate]
    matched_t = {i for i, _ in matches}
    matched_d = {j for _, j in matches}
    unmatched_d = [j for j in range(m) if j not in matched_d]
    unmatched_t = [i for i in range(n) if i not in matched_t]
    return matches, unmatched_d, unmatched_t


def _greedy_from_cost(cost, big_cost):
    n, m = cost.shape
    pairs = sorted(
        ((cost[i, j], i, j) for i in range(n) for j in range(m) if cost[i, j] < big_cost),
        key=lambda t: t[0],
    )
    used_t, used_d, matches = set(), set(), []
    for _, i, j in pairs:
        if i not in used_t and j not in used_d:
            matches.append((i, j))
            used_t.add(i)
            used_d.add(j)
    return matches


def associate_from_cost(cost, big_cost=BIG_COST, greedy=False):
    """Assign rows (tracks) to cols (dets) from a PRECOMPUTED cost matrix. Pairs with
    cost >= big_cost are treated as gated-out. Returns (matches, unmatched_cols, unmatched_rows).
    Used by the KITTI tracker (IoU cost); the synthetic tracker keeps using associate()."""
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(m)), list(range(n))
    if greedy:
        matches = _greedy_from_cost(cost, big_cost)
    else:
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < big_cost]
    matched_t = {i for i, _ in matches}
    matched_d = {j for _, j in matches}
    unmatched_d = [j for j in range(m) if j not in matched_d]
    unmatched_t = [i for i in range(n) if i not in matched_t]
    return matches, unmatched_d, unmatched_t
