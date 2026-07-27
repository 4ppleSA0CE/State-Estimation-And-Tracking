"""associate_from_cost: precomputed cost matrix in, matches/unmatched out; big_cost gates."""
import numpy as np
from association import associate_from_cost, BIG_COST


def test_precomputed_costs_assign():
    cost = np.array([[0.1, BIG_COST], [BIG_COST, 0.2]])
    matches, un_d, un_t = associate_from_cost(cost)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert un_d == [] and un_t == []


def test_gated_pairs_stay_unmatched():
    cost = np.array([[0.1, BIG_COST], [BIG_COST, BIG_COST]])
    matches, un_d, un_t = associate_from_cost(cost)
    assert matches == [(0, 0)]
    assert un_d == [1] and un_t == [1]


def test_empty():
    m, un_d, un_t = associate_from_cost(np.empty((0, 3)))
    assert m == [] and un_d == [0, 1, 2] and un_t == []


def test_greedy_matches_here():
    cost = np.array([[0.1, 0.9], [0.9, 0.2]])
    assert sorted(associate_from_cost(cost, greedy=True)[0]) == [(0, 0), (1, 1)]
