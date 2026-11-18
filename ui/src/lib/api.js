import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  timeout: 120000,
});

// Expected server contract (coordinate order is [latitude, longitude]):
// GET /health -> {status: "ok"}
// GET /trips -> {trips: ["00000000", ...], groups: {student_test: [...], evaluator_test: [...]}}
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
  if (!data.groups || typeof data.groups !== 'object' || Array.isArray(data.groups)) {
    throw new Error('API response must contain trip groups.');
  }
  return Object.fromEntries(
    Object.entries(data.groups).map(([group, trips]) => {
      if (!Array.isArray(trips)) throw new Error(`Trip group "${group}" must be an array.`);
      return [group, trips.map(String)];
    }),
  );
}

export async function predictTrip(tripId) {
  const { data } = await api.post(`/predict/${encodeURIComponent(tripId)}`);
  return data;
}
