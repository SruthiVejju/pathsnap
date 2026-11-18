# UI ↔ FastAPI contract proposal

This is an integration proposal for the UI and backend owners to agree on. It is not a claim about endpoints already implemented.

## Endpoints

- `GET /health` → `{ "status": "ok" }`
- `GET /trips` → `{ "trips": ["<trip-id>", ...], "groups": { "<folder name>": ["<trip-id>", ...] } }`. `trips` is every
  trip the server can score; `groups` splits them by source folder (the server can serve several folders at once via
  `PATHSNAP_EVAL_DIR=dirA:dirB`). Existing clients that read only `trips` keep working.
- `POST /predict/{trip_id}` → prediction response below.

## Prediction response

```json
{
  "trip_id": "<trip-id>",
  "raw_gps": [[56.251559, 54.963519], [56.251523, 54.963143]],
  "predicted_route": [[56.2515, 54.9635], [56.2512, 54.9629]],
  "road_network": [
    [[56.2515, 54.9635], [56.2512, 54.9629]]
  ],
  "summary": {
    "observations": 2,
    "matched_segments": 1,
    "processing_ms": 15.2
  },
  "metrics": {
    "pointwise_accuracy": 0.97,
    "route_continuity": 0.967,
    "route_f1": 0.958,
    "route_precision": 0.961,
    "route_recall": 0.955,
    "pointwise_accuracy_classifier_only": 0.951,
    "route_continuity_classifier_only": 0.827,
    "route_f1_classifier_only": 0.914
  },
  "latency_ms": 599.0
}
```

`metrics` are NUMBERS computed on the server against its private reference data (the brief has the API calculate accuracy and
the UI display it). Definitions: `pointwise_accuracy` = share of GPS points whose predicted arc equals the arc the point aligns
to on the reference route (monotonic alignment, a derived label); `route_continuity` = share of consecutive predicted-arc changes
that follow the directed road network; `route_precision/recall/f1` compare the SET of distinct predicted arcs with the set of
reference-route arcs. `*_classifier_only` is the same score without the HMM decoder, for the comparison. The reference route and
labels themselves are never returned.

All coordinates returned to the browser use `[latitude, longitude]`. Dataset files store longitude first; the server must convert consistently.

## Data-integrity constraints

- The browser sends only the selected trip identifier, not route labels.
- Do not add `reference_route`, ground-truth labels or evaluator `.route` contents to the response (numeric `metrics` are allowed).
- The evaluator's hidden labels remain on Laptop 1, outside the student-side prediction flow and UI.
- `matched_segments` means number of segments in the predicted output, not correctness against hidden reference labels.
- The exact ten trip IDs and their source partition must be confirmed by the backend owner; the UI fetches them dynamically from `/trips`.
