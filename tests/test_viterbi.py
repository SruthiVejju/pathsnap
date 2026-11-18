"""Toy-graph tests for the decoder (no real data needed).

Graph: a straight road 0->1->2->3->4->5 with arcs 0..4 (arc i: node i -> node i+1),
plus a parallel disconnected road with arcs 10..14 (nodes 20..25).
"""
import numpy as np
import pandas as pd

from pathsnap.hmm.graph import RoadGraph
from pathsnap.hmm.metrics import route_continuity
from pathsnap.hmm.viterbi import HMMParams, decode


def toy_graph():
    arcs = np.zeros((15, 2), dtype=int)
    for i in range(5):
        arcs[i] = (i, i + 1)                 # main road
    for i in range(10, 15):
        arcs[i] = (20 + i - 10, 21 + i - 10)  # parallel road, disconnected
    return RoadGraph(arcs)


def make_cands(rows):
    # rows: list of (point_idx, arc_id, prob)
    df = pd.DataFrame(rows, columns=["point_idx", "arc_id", "p"])
    return df[["point_idx", "arc_id"]], df["p"].to_numpy()


def test_viterbi_fixes_disconnected_jump():
    g = toy_graph()
    # True route is arcs 0,1,2,3,4. Classifier prefers the parallel arc 12 at point 2 (wrong, 0.6 vs 0.4).
    rows = []
    for t in range(5):
        rows.append((t, t, 0.8))
        rows.append((t, 10 + t, 0.2))
    rows = [r for r in rows if r[0] != 2] + [(2, 2, 0.4), (2, 12, 0.6)]
    cands, probs = make_cands(rows)
    top1 = cands.assign(p=probs).sort_values("p").groupby("point_idx").tail(1).sort_values("point_idx")
    assert list(top1["arc_id"]) == [0, 1, 12, 3, 4]          # classifier-only breaks the route
    pred = decode(cands, probs, g)
    assert list(pred) == [0, 1, 2, 3, 4]                      # HMM restores it
    assert route_continuity(pred, g) == 1.0
    assert route_continuity(top1.set_index("point_idx")["arc_id"], g) < 1.0


def test_point_without_candidates_is_minus_one():
    g = toy_graph()
    cands, probs = make_cands([(0, 0, 0.9), (2, 2, 0.9)])
    pred = decode(cands, probs, g, n_points=3)
    assert list(pred) == [0, -1, 2]


def test_time_gap_cuts_chain():
    g = toy_graph()
    # Point 1 is far in time; the decoder must not force a transition across the gap.
    cands, probs = make_cands([(0, 0, 0.9), (1, 12, 0.9)])
    pred = decode(cands, probs, g, times=np.array([0.0, 1000.0]), params=HMMParams(gap_s=30))
    assert list(pred) == [0, 12]


def two_way_graph():
    """One two-way road: arcs 0:(0->1) 1:(1->2) forward, arcs 2:(1->0) 3:(2->1) reverse twins."""
    return RoadGraph(np.array([[0, 1], [1, 2], [1, 0], [2, 1]]))


def test_reverse_arc_is_detected():
    g = two_way_graph()
    assert list(g.reverse) == [2, 3, 0, 1]
    assert 2 in g.reachable(0, 2)                               # a free turnaround is reachable by default ...
    assert 2 not in g.reachable(0, 2, no_uturn=True)            # ... but not when U-turns are excluded


def test_uturn_penalty_keeps_direction():
    g = two_way_graph()
    # Vehicle drives forward 0 -> 1. At point 1 the classifier slightly prefers the REVERSE twin of arc 0 (arc 2).
    cands, probs = make_cands([(0, 0, 0.9), (1, 1, 0.45), (1, 2, 0.55), (2, 1, 0.9)])
    free = decode(cands, probs, g, params=HMMParams(hop_penalty=0.1))          # old behaviour: turnaround is cheap
    assert list(free) == [0, 2, 1]
    kept = decode(cands, probs, g, params=HMMParams(hop_penalty=0.1, uturn_penalty=20.0))
    assert list(kept) == [0, 1, 1]
