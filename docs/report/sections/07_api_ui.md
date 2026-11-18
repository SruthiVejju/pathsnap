## 7. API, UI and deployment

**Architecture.** Laptop 1 runs the FastAPI server (`api/main.py`, uvicorn on `0.0.0.0:8000`). Laptop 2 runs the React/Vite UI (`ui/`), which sends only a trip id. For every request the server loads that trip's GPS track and road graph, runs the full pipeline (candidate generation, context features, classifier, HMM/Viterbi) *without* the reference route, then reads the private reference route to compute metrics. Only the predicted route, the raw GPS and numeric metrics are returned: **the reference route and labels never leave the server** (a test, `tests/test_api.py`, checks this).

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, loaded pipeline, trip folders, decoder settings |
| `GET /trips` | trip ids, plus `groups` (ids per source folder) |
| `POST /predict/{trip_id}` (also `POST /predict` with `{"trip_id": ...}`) | predicted route, raw GPS, summary, metrics, latency |

**Metrics returned** (numbers only): `pointwise_accuracy` (share of GPS points whose predicted arc equals the reference-aligned arc), `route_continuity` (share of predicted-arc changes that follow the road network), `route_f1` with `route_precision` and `route_recall` (set of predicted arcs against the set of reference-route arcs), each also for the classifier-only baseline, plus the latency of the pipeline.

**Logging.** Every request is written to the terminal and to `logs/api.log` with the caller's address, e.g.

```
INFO predict trip=<id> points=<n> latency=<ms> ms accuracy=<x> continuity=<x> f1=<x>
INFO POST /predict/<id> from=<Laptop 2 address> -> 200 in <ms> ms
```

**Deployment.** Both laptops on the same network. Laptop 1: `uvicorn api.main:app --host 0.0.0.0 --port 8000` with `PATHSNAP_EVAL_DIR` pointing to the trip folder; Laptop 2: set `VITE_API_BASE_URL=http://<Laptop-1-IP>:8000` in `ui/.env`, then `npm install && npm run dev`. The server refuses to start if the model artifact's feature list does not match the code, so a stale model can never give silently wrong predictions. Full instructions are in the README.

<div class="fig"><img src="figures/ui_student_trip_00000007.jpg" style="width:100%"/><div class="cap">Figure 7.1. The evaluator dashboard on Laptop 2 for student-test trip 00000007: trip selectors (one dropdown per partition), the trip summary, the server-calculated route-quality metrics (accuracy, route continuity, route F1) and the interactive map with the predicted route.</div></div>

<div class="shot" markdown="1">
**SCREENSHOT NEEDED (Subramani B M):** the terminal/CMD of Laptop 1 showing successful `/predict` requests with `from=<Laptop 2's address>` (this proves the request came over the network).
</div>

<div class="team" markdown="1">
**TEAM TO WRITE (Subramani B M and Shubham Singh):** describe how you tested the two-laptop setup (network, firewall, which trips you ran) and anything that failed and how you fixed it.
</div>
