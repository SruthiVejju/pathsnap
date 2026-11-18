"""PathSnap FastAPI backend (runs on Laptop 1).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_DIR  folder containing one sub-folder per trip (.track/.nodes/.arcs/.route).
                         Several folders may be given, separated by the OS path separator (':' on macOS/Linux,
                         ';' on Windows); their trips are listed together.

Ground truth (.route) is read ONLY here, after the prediction is made, to compute metrics.
It is never given to the pipeline.
"""
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.config import ROOT, data_root
from pathsnap.hmm.graph import RoadGraph
from pathsnap.hmm.metrics import pointwise_accuracy_all, route_continuity, route_prf
from pathsnap.pipeline import PlaceholderPipeline, Pipeline, TrainedPipeline

# ---- logging: console + file (evaluators must be shown the API logs) ----
(ROOT / "logs").mkdir(exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
if not log.handlers:   # explicit handlers: basicConfig is a no-op if another lib already configured logging
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / "api.log")):
        _h.setFormatter(_fmt)
        log.addHandler(_h)

_raw_dirs = os.environ.get("PATHSNAP_EVAL_DIR")
EVAL_DIRS = [Path(p) for p in _raw_dirs.split(os.pathsep) if p.strip()] if _raw_dirs else [data_root() / "student_test"]
EVAL_DIR = EVAL_DIRS[0]            # first folder (kept for backwards compatibility)


def _load_pipeline() -> Pipeline:
    """PATHSNAP_PIPELINE=placeholder forces the dev stand-in; otherwise the trained pipeline is
    required and the server FAILS TO START if the artifacts are missing or stale (no silent fallback)."""
    if os.environ.get("PATHSNAP_PIPELINE", "trained") == "placeholder":
        log.warning("Running the DEV-ONLY placeholder pipeline (not the trained model)")
        return PlaceholderPipeline()
    model = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "artifacts" / "candidate_model.joblib"))
    config = Path(os.environ.get("PATHSNAP_MODEL_CONFIG", ROOT / "artifacts" / "model_config.json"))
    pipe = TrainedPipeline(model, config)
    log.info("Loaded trained pipeline from %s", model)
    return pipe


PIPELINE: Pipeline = _load_pipeline()

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
    if trip_id.isalnum():                            # also blocks path traversal
        for base in EVAL_DIRS:
            d = base / trip_id
            if d.is_dir():
                return d
    raise HTTPException(404, f"unknown trip_id {trip_id!r}")


def _polyline(nodes: np.ndarray, arcs: np.ndarray, arc_ids) -> list[list[float]]:
    """Predicted arcs -> [[lat, lon], ...] (UI/Leaflet order), following arc order, repeats collapsed."""
    out, last = [], None
    for a in arc_ids:
        if a < 0 or a == last:
            continue
        last = a
        f, t = nodes[arcs[a, 0]], nodes[arcs[a, 1]]
        start = [float(f[1]), float(f[0])]
        if not out or out[-1] != start:
            out.append(start)
        out.append([float(t[1]), float(t[0])])
    return out


@app.get("/health")
def health():
    params = getattr(PIPELINE, "params", None)
    return {"status": "ok", "eval_dir": str(EVAL_DIR), "eval_dirs": [str(d) for d in EVAL_DIRS], "pipeline": type(PIPELINE).__name__,
            "hmm_params": asdict(params) if params is not None else None}


@app.get("/trips")
def trips():
    """{"trips": [every trip id], "groups": {folder name: [trip ids]}}. A trip id found in several folders is listed once."""
    groups, seen = {}, set()
    for base in EVAL_DIRS:
        ids = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in seen) if base.is_dir() else []
        seen.update(ids)
        groups[base.name] = ids
    return {"trips": sorted(seen), "groups": groups}


def _run(tid: str) -> dict:
    t0 = time.perf_counter()
    d = _trip_dir(tid)
    track = np.loadtxt(d / f"{tid}.track")
    nodes = np.loadtxt(d / f"{tid}.nodes")
    arcs = np.loadtxt(d / f"{tid}.arcs", dtype=int)

    pred = PIPELINE.predict(track, nodes, arcs)          # no ground truth passed in
    latency_ms = (time.perf_counter() - t0) * 1000

    # ---- scoring against the server-side reference; only NUMBERS leave this function ----
    ref = np.loadtxt(d / f"{tid}.route", dtype=int, ndmin=1)
    g = RoadGraph(arcs)

    # pointwise truth = monotonic alignment of the GPS points to the ordered reference route (Part A labels)
    from pathsnap.candidates import align_labels
    from pathsnap.data import load_trip
    truth, _ = align_labels(load_trip(str(d), with_route=True))
    hmm_p, hmm_r, hmm_f1 = route_prf(pred.hmm_arcs, ref)
    clf_p, clf_r, clf_f1 = route_prf(pred.classifier_arcs, ref)

    metrics = {
        # headline numbers
        "pointwise_accuracy": pointwise_accuracy_all(pred.hmm_arcs, truth),
        "route_continuity": route_continuity(pred.hmm_arcs, g),
        "route_f1": hmm_f1,
        "route_precision": hmm_p,
        "route_recall": hmm_r,
        # classifier-only, for the comparison the brief asks for
        "pointwise_accuracy_classifier_only": pointwise_accuracy_all(pred.classifier_arcs, truth),
        "route_continuity_classifier_only": route_continuity(pred.classifier_arcs, g),
        "route_f1_classifier_only": clf_f1,
        # kept for the existing UI/contract
        "pred_arcs_on_reference": hmm_p,
        "pred_arcs_on_reference_classifier_only": clf_p,
        "reference_coverage": hmm_r,
    }
    hmm = pred.hmm_arcs.to_numpy()
    log.info("predict trip=%s points=%d latency=%.0f ms accuracy=%.3f continuity=%.3f f1=%.3f", tid, len(track),
             latency_ms, metrics["pointwise_accuracy"], metrics["route_continuity"], metrics["route_f1"])
    # NOTE: the reference route/labels are deliberately NOT returned (they must never reach the UI).
    return {
        "trip_id": tid,
        "raw_gps": [[float(lat), float(lon)] for lon, lat in track[:, :2]],   # [lat, lon]
        "predicted_route": _polyline(nodes, arcs, hmm),                        # [lat, lon]
        "summary": {
            "observations": int(len(track)),
            "matched_segments": int(len(set(hmm[hmm >= 0].tolist()))),
            "processing_ms": round(latency_ms, 1),
        },
        "metrics": metrics,
        "latency_ms": latency_ms,
    }


@app.post("/predict/{trip_id}")
def predict_by_path(trip_id: str):
    """Endpoint the UI uses: POST /predict/<trip_id>."""
    return _run(trip_id)


@app.post("/predict")
def predict(req: PredictRequest):
    """Same thing with a JSON body {"trip_id": "..."}."""
    return _run(req.trip_id)
