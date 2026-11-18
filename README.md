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
| `configs/config.yaml` | shared constants (seed, CRS, paths) |

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

## Running
_To be filled in as components are built (API, UI, deployment)._

## Team / AI disclosure
_Fill in team members and their areas; disclose AI assistance as per event policy._
