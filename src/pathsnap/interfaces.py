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


# API response contract (FastAPI -> UI). Coordinates are [lat, lon] (Leaflet order).
# POST /predict/{trip_id}  (a JSON-body form POST /predict {"trip_id": ...} also exists) ->
# {
#   "trip_id": str,
#   "raw_gps": [[lat, lon], ...],
#   "predicted_route": [[lat, lon], ...],
#   "summary": {"observations": int, "matched_segments": int, "processing_ms": float},
#   "metrics": {"route_continuity": ..., "route_continuity_classifier_only": ..., ...},  # numbers only
#   "latency_ms": float
# }
# The reference route / labels are NEVER returned: ground truth must not reach the UI.
