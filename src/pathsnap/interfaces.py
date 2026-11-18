"""Agreed interfaces between team members. Change only by team agreement.

Candidate table (one row per GPS point x candidate arc), pandas DataFrame:
    trip_id   : str
    point_idx : int   (0-based row in .track)
    arc_id    : int   (0-based row in that trip's .arcs; NOT comparable across trips)
    <feature columns> : float   (see features/)
    label     : int   (1 if arc is the reference arc for that point, else 0)
                       -- absent at inference time
"""
from typing import Protocol

import numpy as np
import pandas as pd

# Columns that are identifiers/labels, never fed to a model.
ID_COLS = ["trip_id", "point_idx", "arc_id"]
LABEL_COL = "label"


class Classifier(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return array (n, 2); column 1 = P(candidate is the true arc)."""


def decode(candidates: pd.DataFrame, probs: np.ndarray, graph) -> pd.Series:
    """HMM/Viterbi decoder contract.

    candidates: candidate table for ONE trip (no labels)
    probs:      P(true arc) per candidate row, aligned with candidates
    graph:      road network for that trip
    returns:    Series indexed by point_idx -> predicted arc_id
    """
    raise NotImplementedError


# API response contract (FastAPI -> UI)
# POST /predict {"trip_id": "00000000"} ->
# {
#   "trip_id": str,
#   "gps": [[lon, lat], ...],
#   "predicted_route": [[lon, lat], ...],      # polyline of predicted arcs
#   "reference_route": [[lon, lat], ...],      # server-side truth, for display after scoring
#   "metrics": {"pointwise_acc": float, "classifier_only_acc": float, "route_continuity": float},
#   "latency_ms": float
# }
