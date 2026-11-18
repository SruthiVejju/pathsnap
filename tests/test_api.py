"""API smoke test on the student-test partition (never evaluator data)."""
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
    for k in ["gps", "predicted_route", "reference_route", "metrics", "latency_ms"]:
        assert k in body
    assert len(body["predicted_route"]) > 1
    assert 0 <= body["metrics"]["route_continuity"] <= 1


def test_unknown_and_malicious_trip_ids():
    assert client.post("/predict", json={"trip_id": "nope"}).status_code == 404
    assert client.post("/predict", json={"trip_id": "../train"}).status_code == 404
