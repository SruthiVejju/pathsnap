"""Route-level metrics used by the API."""
import numpy as np

from pathsnap.hmm.metrics import pointwise_accuracy_all, route_prf


def test_route_prf_perfect_and_disjoint():
    assert route_prf([1, 2, 3], [3, 2, 1]) == (1.0, 1.0, 1.0)
    assert route_prf([1, 2], [3, 4]) == (0.0, 0.0, 0.0)


def test_route_prf_partial_overlap_is_set_based():
    # predicted {1,2,3,9}: 3 of 4 on the route -> precision .75; reference {1,2,3,4,5,6}: 3 of 6 covered -> recall .5
    p, r, f1 = route_prf([1, 1, 2, 3, 9, -1], [1, 2, 3, 4, 5, 6])
    assert (p, r) == (0.75, 0.5)
    assert abs(f1 - 2 * 0.75 * 0.5 / 1.25) < 1e-12


def test_route_prf_ignores_no_candidate_points_and_empty_input():
    assert route_prf([-1, -1], [1, 2]) == (0.0, 0.0, 0.0)
    assert route_prf([1], []) == (0.0, 0.0, 0.0)


def test_pointwise_accuracy_all_counts_missing_as_wrong():
    assert pointwise_accuracy_all(np.array([1, 2, -1, 4]), np.array([1, 2, 3, 4])) == 0.75
    assert pointwise_accuracy_all(np.array([]), np.array([])) == 0.0
