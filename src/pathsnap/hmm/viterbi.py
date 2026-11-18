"""HMM + Viterbi route decoder.

Model
-----
Hidden state at GPS point t : one of that point's candidate ARCS.
Emission  : log P(arc is true | features), taken from the classifier, times `emission_weight`.
Transition: depends on directed arc-hops h between the previous and the current arc
              h == 0 (same arc)            -> cost 0
              1 <= h <= max_hops (reachable) -> cost  hop_penalty * h
              not reachable                  -> cost  unreachable_penalty   (SOFT, never forbidden,
                                                      so noisy data cannot dead-end the decoder)
            log-transition = -cost.
Viterbi   : delta_t(j) = emit_t(j) + max_i [ delta_{t-1}(i) - cost(i, j) ];  backtrack argmax.

Edge cases
----------
* Time gap > `gap_s` between consecutive points: the chain is cut and each segment is decoded
  independently (no transition is imposed across the gap).
* A GPS point with no candidates is skipped; its output is -1 and the chain bridges over it.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathsnap.hmm.graph import RoadGraph

EPS = 1e-9


@dataclass
class HMMParams:
    emission_weight: float = 1.0
    hop_penalty: float = 1.0
    unreachable_penalty: float = 8.0
    max_hops: int = 6
    gap_s: float = 30.0
    uturn_penalty: float | None = None   # None = old behaviour (a turnaround is an ordinary 1-hop move)


def _cost_matrix(graph: RoadGraph, prev_arcs: np.ndarray, cur_arcs: np.ndarray, p: HMMParams) -> np.ndarray:
    """cost[i, j] of moving from prev_arcs[i] to cur_arcs[j]."""
    cost = np.full((len(prev_arcs), len(cur_arcs)), p.unreachable_penalty, dtype=float)
    no_uturn = p.uturn_penalty is not None
    for i, a in enumerate(prev_arcs):
        reach = graph.reachable(int(a), p.max_hops, no_uturn=no_uturn)
        rev = int(graph.reverse[int(a)])
        for j, b in enumerate(cur_arcs):
            h = reach.get(int(b))
            if h is not None and h <= p.max_hops:   # cache may hold deeper BFS from an earlier call
                cost[i, j] = p.hop_penalty * h
            if no_uturn and int(b) == rev and h != 0:
                cost[i, j] = p.uturn_penalty        # turning around onto the opposite-direction twin
    return cost


def _viterbi_segment(steps, graph: RoadGraph, p: HMMParams) -> list[int]:
    """steps: list of (arcs array, log-emission array). Returns best arc per step."""
    delta = steps[0][1].copy()
    back = []
    for t in range(1, len(steps)):
        arcs_prev, arcs_cur, emit = steps[t - 1][0], steps[t][0], steps[t][1]
        cost = _cost_matrix(graph, arcs_prev, arcs_cur, p)
        scores = delta[:, None] - cost                      # (prev, cur)
        bp = scores.argmax(axis=0)
        delta = scores[bp, np.arange(len(arcs_cur))] + emit
        back.append(bp)
    k = int(delta.argmax())
    path = [k]
    for bp in reversed(back):
        k = int(bp[k])
        path.append(k)
    path.reverse()
    return [int(steps[t][0][k]) for t, k in enumerate(path)]


def decode(
    candidates: pd.DataFrame,
    probs: np.ndarray,
    graph: RoadGraph,
    times: np.ndarray | None = None,
    params: HMMParams | None = None,
    n_points: int | None = None,
) -> pd.Series:
    """Decode one trip.

    candidates : DataFrame with columns point_idx, arc_id (one row per point x candidate)
    probs      : classifier P(true arc), aligned with candidates rows
    graph      : RoadGraph of that trip
    times      : optional timestamps (seconds) indexed by point_idx, used for gap handling
    n_points   : total number of GPS points (so points with no candidates appear as -1)
    returns    : Series indexed by point_idx -> predicted arc_id (-1 if no candidates)
    """
    p = params or HMMParams()
    df = pd.DataFrame({
        "point_idx": candidates["point_idx"].to_numpy(),
        "arc_id": candidates["arc_id"].to_numpy(),
        "logp": np.log(np.clip(np.asarray(probs, dtype=float), EPS, 1.0)) * p.emission_weight,
    })
    out_len = n_points if n_points is not None else int(df["point_idx"].max()) + 1
    result = pd.Series(-1, index=np.arange(out_len), dtype=int)

    pidx, steps = [], []
    for pi, g in df.groupby("point_idx", sort=True):
        pidx.append(int(pi))
        steps.append((g["arc_id"].to_numpy(), g["logp"].to_numpy()))

    # Cut into segments at large time gaps.
    cuts = [0]
    if times is not None:
        t = np.asarray(times, dtype=float)
        for k in range(1, len(pidx)):
            if t[pidx[k]] - t[pidx[k - 1]] > p.gap_s:
                cuts.append(k)
    cuts.append(len(pidx))

    for s, e in zip(cuts[:-1], cuts[1:]):
        if e - s == 0:
            continue
        path = _viterbi_segment(steps[s:e], graph, p)
        for pi, arc in zip(pidx[s:e], path):
            result.iloc[pi] = arc
    return result
