"""Route-level metrics that do not need per-point labels."""
import numpy as np
import pandas as pd

from pathsnap.hmm.graph import RoadGraph


def route_continuity(pred: pd.Series, graph: RoadGraph, max_hops: int = 1) -> float:
    """Fraction of consecutive predicted-arc changes that are physically connected.

    Repeats of the same arc are ignored (no change). Points with arc -1 are skipped.
    1.0 means every move from one arc to a different arc follows the directed network
    within `max_hops` arc-steps. A prediction with no changes returns 1.0.
    """
    arcs = [int(a) for a in pred.to_numpy() if a >= 0]
    changes = [(a, b) for a, b in zip(arcs[:-1], arcs[1:]) if a != b]
    if not changes:
        return 1.0
    ok = sum(graph.hops(a, b, max_hops) is not None for a, b in changes)
    return ok / len(changes)


def pointwise_accuracy(pred: pd.Series, truth: pd.Series) -> float:
    """Share of points whose predicted arc equals the reference arc (needs per-point truth)."""
    m = truth >= 0
    return float((pred[m] == truth[m]).mean())


def route_prf(pred_arcs, reference_arcs) -> tuple[float, float, float]:
    """Route-level precision / recall / F1 on the SET of directed arcs.

    precision = share of distinct predicted arcs that are on the reference route,
    recall    = share of distinct reference arcs that the prediction covers,
    F1        = their harmonic mean (0 when either is 0).
    Arc -1 (no candidate) is ignored. Needs only the reference route, not per-point labels.
    """
    pred = {int(a) for a in np.asarray(pred_arcs) if a >= 0}
    ref = {int(a) for a in np.asarray(reference_arcs)}
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    hit = len(pred & ref)
    p, r = hit / len(pred), hit / len(ref)
    return p, r, (2 * p * r / (p + r) if hit else 0.0)


def pointwise_accuracy_all(pred, truth) -> float:
    """Share of ALL GPS points whose predicted arc equals the reference-aligned arc (-1 predictions count as wrong)."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    return float((pred == truth).mean()) if len(truth) else 0.0
