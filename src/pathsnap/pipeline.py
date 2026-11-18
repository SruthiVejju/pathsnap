"""Inference pipeline contract used by the API.

The API only ever calls `Pipeline.predict(track, nodes, arcs)`. It never passes reference
routes into the pipeline (ground truth stays server-side).

`PlaceholderPipeline` is a DEV-ONLY stand-in: nearest-arc distance scoring + the real HMM
decoder. Replace it with the team's trained pipeline (candidate generation -> features ->
trained classifier -> decode) by implementing the same `predict` signature.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pathsnap.data.geo import make_projector, project
from pathsnap.hmm.graph import RoadGraph
from pathsnap.hmm.viterbi import HMMParams, decode


@dataclass
class Prediction:
    hmm_arcs: pd.Series          # point_idx -> arc_id after HMM/Viterbi
    classifier_arcs: pd.Series   # point_idx -> arc_id, top-1 classifier only (for comparison)


class Pipeline(Protocol):
    def predict(self, track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray) -> Prediction: ...


class TrainedPipeline:
    """Real pipeline: shared candidate generator -> trained classifier -> HMM/Viterbi.

    Uses pathsnap.candidates.build_candidates (never sees the route) and the exact feature list the
    model was trained on. Refuses to load a model whose saved feature list differs from
    FEATURE_COLUMNS, so a stale artifact can never silently produce wrong predictions.
    """

    def __init__(self, model_path, config_path, params: HMMParams | None = None):
        import json

        import joblib

        from pathsnap.candidates import FEATURE_COLUMNS

        cfg = json.loads(open(config_path).read())
        saved = cfg.get("feature_columns") or cfg.get("features")
        if saved != list(FEATURE_COLUMNS):
            raise RuntimeError(
                f"Model/config feature list {saved} does not match pathsnap.candidates.FEATURE_COLUMNS "
                f"{list(FEATURE_COLUMNS)}. Retrain the model with the shared generator.")
        self.model = joblib.load(model_path)
        self.params = params or HMMParams()
        self.features = list(FEATURE_COLUMNS)

    def predict(self, track, nodes, arcs) -> Prediction:
        from pathsnap.candidates import build_candidates
        from pathsnap.data import Trip

        tf = make_projector(track[:, :2])
        trip = Trip("request", track, nodes, arcs, None, project(tf, track[:, :2]), project(tf, nodes[:, :2]))
        n = len(track)
        df = build_candidates(trip)                       # no route is available here
        if df.empty:
            empty = pd.Series(-1, index=np.arange(n), dtype=int)
            return Prediction(empty, empty.copy())

        probs = self.model.predict_proba(df[self.features].to_numpy(dtype=np.float32))[:, 1]
        top = df.assign(p=probs).sort_values("p").groupby("point_idx").tail(1).set_index("point_idx")["arc_id"]
        top1 = top.reindex(range(n), fill_value=-1).astype(int)
        hmm = decode(df[["point_idx", "arc_id"]], probs, RoadGraph(arcs),
                     times=track[:, 2], params=self.params, n_points=n)
        return Prediction(hmm, top1)


class PlaceholderPipeline:
    """DEV ONLY. Distance-based score instead of a trained model."""

    def __init__(self, radius_m: float = 60.0, k: int = 12, scale_m: float = 15.0, params: HMMParams | None = None):
        self.radius_m, self.k, self.scale_m = radius_m, k, scale_m
        self.params = params or HMMParams()

    def predict(self, track, nodes, arcs) -> Prediction:
        tf = make_projector(track[:, :2])
        N, P = project(tf, nodes), project(tf, track[:, :2])
        A, B = N[arcs[:, 0]], N[arcs[:, 1]]
        tree = cKDTree((A + B) / 2)
        _, near = tree.query(P, k=min(self.k * 3, len(arcs)))   # shortlist by midpoint, then exact distance

        rows = []
        for i, cand in enumerate(near):
            a, b = A[cand], B[cand]
            ab = b - a
            l2 = np.maximum((ab ** 2).sum(1), 1e-12)
            t = np.clip(((P[i] - a) * ab).sum(1) / l2, 0, 1)
            d = np.linalg.norm(P[i] - (a + t[:, None] * ab), axis=1)
            keep = np.argsort(d)[: self.k]
            keep = keep[d[keep] <= self.radius_m]
            rows += [(i, int(cand[j]), float(d[j])) for j in keep]

        n = len(P)
        if not rows:
            empty = pd.Series(-1, index=np.arange(n), dtype=int)
            return Prediction(empty, empty.copy())
        c = pd.DataFrame(rows, columns=["point_idx", "arc_id", "dist"])
        c["p"] = np.exp(-c["dist"] / self.scale_m)
        c["p"] /= c.groupby("point_idx")["p"].transform("sum")

        top1 = c.sort_values("p").groupby("point_idx").tail(1).set_index("point_idx")["arc_id"]
        top1 = top1.reindex(range(n), fill_value=-1).astype(int)
        hmm = decode(c[["point_idx", "arc_id"]], c["p"].to_numpy(), RoadGraph(arcs),
                     times=track[:, 2], params=self.params, n_points=n)
        return Prediction(hmm, top1)
