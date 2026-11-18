# UI ↔ FastAPI contract proposal

This is an integration proposal for the UI and backend owners to agree on. It is not a claim about endpoints already implemented.

## Endpoints

- `GET /health` → `{ "status": "ok" }`
- `GET /trips` → `{ "trips": ["<permitted-trip-id>", ...] }`
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
  }
}
```

All coordinates returned to the browser use `[latitude, longitude]`. Dataset files store longitude first; the server must convert consistently.

## Data-integrity constraints

- The browser sends only the selected trip identifier, not route labels.
- Do not add `reference_route`, ground-truth labels, evaluator scores or evaluator `.route` contents to the response.
- The evaluator's hidden labels remain on Laptop 1, outside the student-side prediction flow and UI.
- `matched_segments` means number of segments in the predicted output, not correctness against hidden reference labels.
- The exact ten trip IDs and their source partition must be confirmed by the backend owner; the UI fetches them dynamically from `/trips`.
