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

## Data and feature pipeline (Part A)

`src/pathsnap/candidates.py` generates up to 25 road-arc candidates within 60 m
for every GPS point, without reading the reference route. Its `FEATURE_COLUMNS`
list is the canonical ordered list of 11 model features. `add_labels` is
separate and uses the ordered reference route only to generate monotonic
training labels. Heading and speed features do not bridge GPS gaps over five
seconds.

From the repository root in PowerShell, with the dataset in the sibling
`Student folder` directory:

```powershell
.\.venv\Scripts\python.exe scripts\eda.py '..\Student folder' results\eda
.\.venv\Scripts\python.exe scripts\build_dataset.py '..\Student folder\train' results\train.csv.gz
.\.venv\Scripts\python.exe scripts\build_dataset.py '..\Student folder\student_test' results\student_test.csv.gz
```

The candidate table has `trip_id`, `point_idx`, `arc_id`, the 11 columns in
`FEATURE_COLUMNS`, and `label`. The `*_miss.csv` file records per-trip candidate
miss rates and aligned-route distances. Evaluator-testing data and labels must
not be used during development.

To create a sensitivity table excluding flagged training trips, while keeping
their audit metrics and reasons:

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py '..\Student folder\train' results\train_exclude_33_54.csv.gz --exclude-trip 00000033 --exclude-trip 00000054
```

## Decoder tuning and map

`scripts/tune_decoder.py` tunes the HMM/Viterbi settings with Bayesian optimisation (Optuna TPE) on the 80 training
trips using out-of-fold classifier probabilities, then compares classifier-only / default HMM / tuned HMM on the training
and student-test trips. Results and the run history are in `results/decoder_tuning/`; the tuned settings are saved to
`artifacts/hmm_config.json` and loaded automatically by the server (shown on `/health`).

`scripts/folium_prediction_map.py <trip_dir> <out.html>` runs the real pipeline and writes an interactive map with raw GPS,
road network, reference route, classifier+HMM route and classifier-only route (layer control + legend). Examples are in `maps/`.

## Context features, final classifier and decoder (Part B/C)

`src/pathsnap/context.py` adds 12 neighbour-point features on top of the 11 base features (`EXTENDED_COLUMNS`, 23 in total).
They describe the *same candidate arc* as seen from the previous/next one or two GPS points: its distance there, how far the
vehicle moved *along* the arc (forward on the true arc, backward on its opposite-direction twin), how many neighbouring points
support the arc, a windowed distance rank, and a windowed heading. Like `build_candidates` it never reads the route.

The decoder has an optional U-turn penalty (`HMMParams.uturn_penalty`): a turnaround onto the opposite-direction twin arc is no
longer a free one-hop move. `scripts/train_context_model.py` produces the final artifacts in one reproducible run (seed 42):
out-of-fold probabilities on the 80 training trips, Bayesian (Optuna TPE) decoder tuning on them, the final classifier fitted on
all training rows, and a comparison on the student-test trips. Logs and per-trip tables are in `results/context_model/`.

| Pointwise accuracy on covered points | Classifier only | HMM (default) | HMM (tuned) |
|---|---|---|---|
| 80 training trips (out-of-fold) | 89.9% | 92.2% | **95.8%** |
| 10 student-test trips | 90.9% | 92.9% | **96.5%** |

Pointwise "truth" is the monotonic route alignment (a derived label). The model config lists its feature set; the server builds
context features only for a model that was trained with them and refuses a model whose feature list is unknown.

## Running the API server (Laptop 1)

The API hosts the pipeline (candidates -> classifier -> HMM/Viterbi) and scores predictions
against reference routes that stay **server-side**. The UI (Laptop 2) only sends a trip ID.

### 1. Point the server at the trips
Each trip is a folder `<trip_id>/` containing `<trip_id>.track`, `.nodes`, `.arcs` and `.route`.
```bash
export PATHSNAP_EVAL_DIR="/path/to/folder/with/trip/subfolders"
# several folders at once (':' on macOS/Linux, ';' on Windows), e.g. development + evaluator trips:
export PATHSNAP_EVAL_DIR="/path/to/student_test:/path/to/evaluator_test"
```
If unset, it defaults to `<PATHSNAP_DATA>/student_test` (development data).
The `.route` file is read only after prediction, to compute metrics; it is never given to the pipeline.

### 2. Start it
```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
`--host 0.0.0.0` is required so the other laptop can reach it (the default `127.0.0.1` is local-only).
Startup loads the model and can take about 10 seconds; wait for `Application startup complete` before opening the UI.

### 3. Find Laptop 1's address and allow the port
- macOS: `ipconfig getifaddr en0` (Wi-Fi). Windows: `ipconfig` -> IPv4 Address. Linux: `hostname -I`.
- Both laptops must be on the same network. If the connection is refused, allow incoming connections
  for Python/port 8000 in the OS firewall.
- The UI should call `http://<laptop-1-ip>:8000`.

### 4. Check it from Laptop 2 (before opening the UI)
```bash
curl http://<laptop-1-ip>:8000/health
curl http://<laptop-1-ip>:8000/trips
curl -X POST http://<laptop-1-ip>:8000/predict/<a trip id>
```

### Endpoints
| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status, eval directory, which pipeline is loaded |
| GET | `/trips` | list of available trip IDs |
| POST | `/predict/{trip_id}` (also `/predict` with `{"trip_id": "..."}`) | `raw_gps`, `predicted_route` (both `[lat, lon]`), `summary`, `metrics`, `latency_ms`. The reference route is never returned. |

Metrics (numbers only) include `pointwise_accuracy`, `route_continuity`, `route_f1` (with precision and recall),
each also for the classifier-only baseline, plus the older `pred_arcs_on_reference` / `reference_coverage`.
Definitions are in `docs/ui-api-contract.md`. `GET /trips` returns `trips` and a `groups` map by source folder. Interactive docs are at `http://<laptop-1-ip>:8000/docs`.

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
By default the server loads the **trained pipeline** (`TrainedPipeline` in `src/pathsnap/pipeline.py`):
`build_candidates` (shared generator, never sees the route) -> trained classifier -> HMM/Viterbi.
It reads `artifacts/candidate_model.joblib` and `artifacts/model_config.json` (override with
`PATHSNAP_MODEL` / `PATHSNAP_MODEL_CONFIG`) and **refuses to start** if the saved feature list differs
from `pathsnap.candidates.FEATURE_COLUMNS` (a stale model cannot silently give wrong predictions).
`/health` reports which pipeline is loaded.

`PATHSNAP_PIPELINE=placeholder` selects a dev-only stand-in (nearest-arc scoring + the real decoder);
it is used by the unit tests and must not be used for the final evaluation.

## Credits and provenance

This repository is a clean-history copy of the PathSnap team project (original team repository: SruthiVejju/pathsnap).
Its commit history was squashed, so authorship is recorded here instead:

| Part | Author (team repo) |
|---|---|
| Data loading, candidate generation, feature engineering, label alignment, EDA (`src/pathsnap/candidates.py`, `src/pathsnap/data/__init__.py`, `scripts/build_dataset.py`, `scripts/eda.py`, `scripts/folium_map.py`, `tests/test_data_features.py`) | Shwetha S |
| Model experiments and Bayesian tuning (`training/`, `README-ML.md`) | Sruthi |
| Evaluator UI and EDA plots (`ui/`, `scripts/eda_plots.py`, `maps/map_preview.py`) | Shubham Singh |
| HMM/Viterbi decoder, FastAPI backend, decoder tuning, Folium prediction maps, per-trip projection helper, data-quality note (`src/pathsnap/hmm/`, `src/pathsnap/pipeline.py`, `api/`, `scripts/tune_decoder.py`, `scripts/folium_prediction_map.py`, `src/pathsnap/data/geo.py`, `docs/`) | SubramaniBM, with AI assistance (Claude Code) |

The model in `artifacts/` was retrained by re-running the team's unmodified `training/experiments.py` on the Part A candidate table
(`results/comparison.csv`); the decoder settings in `artifacts/hmm_config.json` come from `results/decoder_tuning/`.

## Team / AI disclosure

| Team member | Area of contribution |
|---|---|
| **Sruthi Vejju** | Model training and comparison, trip-grouped cross-validation, Optuna hyperparameter tuning, and evaluation before and after HMM decoding |
| **Shwetha S** | Data exploration, GPS preprocessing, road candidate generation, feature engineering, and reference-label alignment |
| **Subramani B M** | FastAPI backend, HMM/Viterbi decoder, and integration of the trained model |
| **Shubham Singh** | Evaluator UI, Folium maps, and project visualisations |

**AI assistance**- We used ChatGPT and Claude for general project support.
