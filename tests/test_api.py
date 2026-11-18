"""API smoke test on the student-test partition (never evaluator data)."""
import os

os.environ["PATHSNAP_PIPELINE"] = "placeholder"   # tests must not depend on trained artifacts

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_and_trips():
    assert client.get("/health").json()["status"] == "ok"
    trips = client.get("/trips").json()["trips"]
    assert len(trips) == 10


def test_predict_shape_and_no_truth_leak_in_pipeline():
    tid = client.get("/trips").json()["trips"][0]
    r = client.post("/predict", json={"trip_id": tid})
    assert r.status_code == 200
    body = r.json()
    for k in ["raw_gps", "predicted_route", "summary", "metrics", "latency_ms"]:
        assert k in body
    assert "reference_route" not in body and "gps" not in body          # ground truth never sent to the UI
    import numpy as np
    from api.main import EVAL_DIR
    track = np.loadtxt(EVAL_DIR / tid / f"{tid}.track")                 # file is lon, lat, t
    assert body["raw_gps"][0] == [track[0, 1], track[0, 0]]              # API sends [lat, lon], exactly
    r2 = client.post(f"/predict/{tid}")                                  # path form used by the UI
    assert r2.status_code == 200 and r2.json()["trip_id"] == tid
    assert r2.json()["summary"]["observations"] == len(r2.json()["raw_gps"])
    assert len(body["predicted_route"]) > 1
    assert 0 <= body["metrics"]["route_continuity"] <= 1


def test_unknown_and_malicious_trip_ids():
    assert client.post("/predict", json={"trip_id": "nope"}).status_code == 404
    assert client.post("/predict", json={"trip_id": "../train"}).status_code == 404


def test_trained_pipeline_rejects_stale_feature_list(tmp_path):
    import json

    import joblib
    import pytest

    from pathsnap.pipeline import TrainedPipeline

    joblib.dump({"dummy": 1}, tmp_path / "m.joblib")
    (tmp_path / "c.json").write_text(json.dumps({"features": ["distance_m", "rank"]}))
    with pytest.raises(RuntimeError, match="FEATURE_COLUMNS"):
        TrainedPipeline(tmp_path / "m.joblib", tmp_path / "c.json")


def test_trained_pipeline_loads_tuned_hmm_params(tmp_path):
    import json

    import joblib

    from pathsnap.candidates import FEATURE_COLUMNS
    from pathsnap.pipeline import TrainedPipeline

    joblib.dump({"dummy": 1}, tmp_path / "m.joblib")
    (tmp_path / "c.json").write_text(json.dumps({"feature_columns": list(FEATURE_COLUMNS)}))
    assert TrainedPipeline(tmp_path / "m.joblib", tmp_path / "c.json").params.hop_penalty == 1.0   # defaults
    (tmp_path / "hmm_config.json").write_text(json.dumps({"params": {"hop_penalty": 0.5, "max_hops": 9}}))
    p = TrainedPipeline(tmp_path / "m.joblib", tmp_path / "c.json").params
    assert p.hop_penalty == 0.5 and p.max_hops == 9


def test_metrics_keys_and_ranges():
    tid = client.get("/trips").json()["trips"][0]
    m = client.post(f"/predict/{tid}").json()["metrics"]
    for k in ["pointwise_accuracy", "route_continuity", "route_f1", "route_precision", "route_recall",
              "pointwise_accuracy_classifier_only", "route_f1_classifier_only"]:
        assert 0.0 <= m[k] <= 1.0, k
    # F1 is the harmonic mean of precision and recall
    p, r = m["route_precision"], m["route_recall"]
    assert abs(m["route_f1"] - (2 * p * r / (p + r) if p + r else 0.0)) < 1e-9
    assert all(isinstance(v, float) for v in m.values())                # numbers only, never labels


def test_several_eval_folders_are_listed_together(tmp_path, monkeypatch):
    import api.main as main
    empty = tmp_path / "other_partition"
    empty.mkdir()
    monkeypatch.setattr(main, "EVAL_DIRS", [empty, main.EVAL_DIR])
    body = client.get("/trips").json()
    assert len(body["trips"]) == 10 and body["groups"]["other_partition"] == []
    assert len(body["groups"][main.EVAL_DIR.name]) == 10
    assert client.post(f"/predict/{body['trips'][0]}").status_code == 200   # found in the second folder
    assert client.post("/predict/nope").status_code == 404
