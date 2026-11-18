"""PathSnap FastAPI backend (runs on Laptop 1).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR  folder containing one sub-folder per trip (.track/.nodes/.arcs/.route)

Ground truth (.route) is read ONLY here, after the prediction is made, to compute metrics.
It is never given to the pipeline.
"""
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.config import ROOT, data_root
from pathsnap.hmm.graph import RoadGraph
from pathsnap.hmm.metrics import route_continuity
from pathsnap.pipeline import PlaceholderPipeline, Pipeline

# ---- logging: console + file (evaluators must be shown the API logs) ----
(ROOT / "logs").mkdir(exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
if not log.handlers:   # explicit handlers: basicConfig is a no-op if another lib already configured logging
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / "api.log")):
        _h.setFormatter(_fmt)
        log.addHandler(_h)

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", data_root() / "student_test"))
PIPELINE: Pipeline = PlaceholderPipeline()   # TODO: swap for the trained pipeline

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info("%s %s from=%s -> %s in %.1f ms", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, (time.perf_counter() - t0) * 1000)
    return resp


class PredictRequest(BaseModel):
    trip_id: str


def _trip_dir(trip_id: str) -> Path:
    d = EVAL_DIR / trip_id
    if not trip_id.isalnum() or not d.is_dir():      # also blocks path traversal
        raise HTTPException(404, f"unknown trip_id {trip_id!r}")
    return d


def _polyline(nodes: np.ndarray, arcs: np.ndarray, arc_ids) -> list[list[float]]:
    """Predicted/reference arcs -> [[lon, lat], ...] following the arc order (repeats collapsed)."""
    out, last = [], None
    for a in arc_ids:
        if a < 0 or a == last:
            continue
        last = a
        f, t = nodes[arcs[a, 0]], nodes[arcs[a, 1]]
        if not out or out[-1] != [float(f[0]), float(f[1])]:
            out.append([float(f[0]), float(f[1])])
        out.append([float(t[0]), float(t[1])])
    return out


@app.get("/health")
def health():
    return {"status": "ok", "eval_dir": str(EVAL_DIR), "pipeline": type(PIPELINE).__name__}


@app.get("/trips")
def trips():
    return {"trips": sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir())}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    d = _trip_dir(req.trip_id)
    tid = req.trip_id
    track = np.loadtxt(d / f"{tid}.track")
    nodes = np.loadtxt(d / f"{tid}.nodes")
    arcs = np.loadtxt(d / f"{tid}.arcs", dtype=int)

    pred = PIPELINE.predict(track, nodes, arcs)          # no ground truth passed in
    latency_ms = (time.perf_counter() - t0) * 1000

    # ---- scoring against server-side reference ----
    ref = np.loadtxt(d / f"{tid}.route", dtype=int, ndmin=1)
    g = RoadGraph(arcs)
    ref_set = set(ref.tolist())
    def on_ref(s):
        v = [a for a in s if a >= 0]
        return float(np.mean([a in ref_set for a in v])) if v else 0.0
    def ref_cov(s):
        return float(len(ref_set & set(s[s >= 0].tolist())) / len(ref_set))
    metrics = {
        "route_continuity": route_continuity(pred.hmm_arcs, g),
        "route_continuity_classifier_only": route_continuity(pred.classifier_arcs, g),
        "pred_arcs_on_reference": on_ref(pred.hmm_arcs),
        "pred_arcs_on_reference_classifier_only": on_ref(pred.classifier_arcs),
        "reference_coverage": ref_cov(pred.hmm_arcs),
    }
    log.info("predict trip=%s points=%d latency=%.0f ms continuity=%.3f", tid, len(track), latency_ms,
             metrics["route_continuity"])
    return {
        "trip_id": tid,
        "gps": track[:, :2].tolist(),
        "predicted_route": _polyline(nodes, arcs, pred.hmm_arcs.to_numpy()),
        "reference_route": _polyline(nodes, arcs, ref),
        "metrics": metrics,
        "latency_ms": latency_ms,
    }
