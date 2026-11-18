# PathSnap: Distributed Map-Matching & Route Reconstruction

PES University, UE24CS352A Machine Learning hackathon.
Noisy GPS tracks -> candidate road segments -> classifier -> HMM/Viterbi -> reconstructed route,
served via FastAPI (laptop 1) and a UI (laptop 2).

## Layout
| Path | Purpose |
|---|---|
| `src/pathsnap/data` | loading, cleaning, graph, candidate generation, labels |
| `src/pathsnap/features` | feature engineering |
| `src/pathsnap/models` | model training, CV, evaluation, Optuna tuning |
| `src/pathsnap/hmm` | HMM + Viterbi decoder |
| `api/` | FastAPI backend |
| `ui/` | evaluator-facing UI |
| `notebooks/` | EDA and experiments |
| `maps/` | Folium HTML maps and map-generation code |
| `results/` | experiment logs, CSV summaries |
| `artifacts/` | exported model + config for inference |
| `configs/config.yaml` | shared constants (seed, paths); projection is per-trip, see `src/pathsnap/data/geo.py` |

## Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `import pathsnap` work
export PATHSNAP_DATA="/path/to/Student folder"   # contains train/ and student_test/
```

## Rules we follow
- Evaluator trips are never used in development code.
- Seed 42 everywhere; CV is GroupKFold by trip.
- Ground-truth labels stay server-side in the API.

## Running the API server (Laptop 1)

The API hosts the pipeline (candidates -> classifier -> HMM/Viterbi) and scores predictions
against reference routes that stay **server-side**. The UI (Laptop 2) only sends a trip ID.

### 1. Point the server at the trips
Each trip is a folder `<trip_id>/` containing `<trip_id>.track`, `.nodes`, `.arcs` and `.route`.
```bash
export PATHSNAP_EVAL_DIR="/path/to/folder/with/trip/subfolders"
```
If unset, it defaults to `<PATHSNAP_DATA>/student_test` (development data).
The `.route` file is read only after prediction, to compute metrics; it is never given to the pipeline.

### 2. Start it
```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
`--host 0.0.0.0` is required so the other laptop can reach it (the default `127.0.0.1` is local-only).

### 3. Find Laptop 1's address and allow the port
- macOS: `ipconfig getifaddr en0` (Wi-Fi). Windows: `ipconfig` -> IPv4 Address. Linux: `hostname -I`.
- Both laptops must be on the same network. If the connection is refused, allow incoming connections
  for Python/port 8000 in the OS firewall.
- The UI should call `http://<laptop-1-ip>:8000`.

### 4. Check it from Laptop 2 (before opening the UI)
```bash
curl http://<laptop-1-ip>:8000/health
curl http://<laptop-1-ip>:8000/trips
curl -X POST http://<laptop-1-ip>:8000/predict -H "Content-Type: application/json" -d '{"trip_id":"<a trip id>"}'
```

### Endpoints
| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status, eval directory, which pipeline is loaded |
| GET | `/trips` | list of available trip IDs |
| POST | `/predict` `{"trip_id": "..."}` | `gps`, `predicted_route`, `reference_route` (as `[lon, lat]` lists), `metrics`, `latency_ms` |

Metrics include route continuity (HMM vs classifier-only), the share of predicted arcs on the
reference route, and reference coverage. Interactive docs are at `http://<laptop-1-ip>:8000/docs`.

### Logs (show these to the evaluator)
Every request is logged to the terminal and to `logs/api.log`, for example:
```
INFO predict trip=00000000 points=2503 latency=120 ms continuity=0.929
INFO POST /predict from=192.168.1.23 -> 200 in 127.1 ms
```
The `from=` field shows the UI laptop's address, which demonstrates the request came over the network.

### Tests
```bash
python -m pytest tests -q
```

### Status
The server currently runs a **dev-only placeholder** pipeline (`PlaceholderPipeline` in
`src/pathsnap/pipeline.py`: nearest-arc scoring + the real HMM decoder). It must be replaced by the
trained pipeline before the final evaluation; `/health` reports which pipeline is loaded.

## Team / AI disclosure
_Fill in team members and their areas; disclose AI assistance as per event policy._
