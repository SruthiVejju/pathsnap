"""Context (neighbour-point) features: correctness on a toy trip and no dependence on the route."""
import numpy as np
import pandas as pd

from pathsnap.candidates import build_candidates
from pathsnap.context import CONTEXT_COLUMNS, EXTENDED_COLUMNS, add_context
from pathsnap.data import Trip


def straight_trip(n=8, step=10.0, gap_at=None):
    """Vehicle moving +x at 10 m/s along a road made of 3 arcs of 30 m each (plus the reverse twins)."""
    nodes = np.array([[0, 0], [30, 0], [60, 0], [90, 0]], dtype=float)
    arcs = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2]])
    pxy = np.array([[5.0 + step * i, 3.0] for i in range(n)])
    t = np.arange(n, dtype=float)
    if gap_at is not None:
        t[gap_at:] += 100.0
    track = np.c_[pxy, t]
    return Trip("toy", track, nodes, arcs, None, pxy, nodes)


def table(trip):
    return add_context(build_candidates(trip), trip)


def test_columns_present_and_extended_list():
    df = table(straight_trip())
    assert EXTENDED_COLUMNS[-len(CONTEXT_COLUMNS):] == CONTEXT_COLUMNS
    assert set(EXTENDED_COLUMNS) <= set(df.columns)
    assert len(df) == len(build_candidates(straight_trip()))             # no rows added or lost


def test_forward_arc_moves_forward_reverse_arc_moves_backward():
    df = table(straight_trip())
    row = lambda i, a: df[(df.point_idx == i) & (df.arc_id == a)].iloc[0]
    fwd, rev = row(3, 1), row(3, 4)                                      # arc 1 = 1->2, arc 4 = 2->1 (its twin)
    assert fwd.along_prev1_m > 0 and fwd.along_next1_m > 0               # frac_along grows along the travel direction
    assert rev.along_prev1_m < 0 and rev.along_next1_m < 0               # the twin's frac_along shrinks
    assert fwd.cos_heading_win > 0.9 and rev.cos_heading_win < -0.9


def test_absent_neighbour_is_nan_and_time_gap_is_ignored():
    df = table(straight_trip(gap_at=4))
    first = df[df.point_idx == 0]
    assert first["dist_prev1"].isna().all()                              # nothing before point 0
    across = df[(df.point_idx == 4)]                                     # previous point is 100 s earlier
    assert across["dist_prev1"].isna().all() and across["along_prev1_m"].isna().all()


def test_context_never_needs_a_route():
    trip = straight_trip()
    assert trip.route is None
    add_context(build_candidates(trip), trip)                            # must not raise without a route


def test_empty_table_is_handled():
    trip = straight_trip()
    out = add_context(pd.DataFrame(columns=["point_idx", "arc_id"]), trip)
    assert out.empty and set(CONTEXT_COLUMNS) <= set(out.columns)
