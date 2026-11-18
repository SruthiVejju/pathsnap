import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  timeout: 120000,
});

// Expected server contract (coordinate order is [latitude, longitude]):
// GET /health -> {status: "ok"}
// GET /trips -> {trips: ["00000000", ...]}
// POST /predict/{tripId} -> {
//   trip_id, raw_gps: [[lat, lon], ...],
//   predicted_route: [[lat, lon], ...],
//   road_network: [[[lat, lon], [lat, lon]], ...] (optional),
//   summary: {observations, matched_segments, processing_ms} (optional),
//   metrics: {pointwise_accuracy, route_continuity, route_f1} (optional)
// }
// Reference labels/routes must not be returned by evaluator endpoints.
export async function getHealth() {
  const { data } = await api.get('/health');
  return data;
}

export async function getTrips() {
  const { data } = await api.get('/trips');
  if (!Array.isArray(data.trips)) throw new Error('API response must contain a trips array.');
  return data.trips.map(String);
}

export async function predictTrip(tripId) {
  const { data } = await api.post(`/predict/${encodeURIComponent(tripId)}`);
  return data;
}
