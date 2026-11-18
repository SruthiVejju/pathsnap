import { useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, useMap } from 'react-leaflet';

function FitBounds({ rawGps, predictedRoute }) {
  const map = useMap();
  useEffect(() => {
    const points = [...(rawGps || []), ...(predictedRoute || [])]
      .filter((p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite));
    if (points.length) map.fitBounds(points, { padding: [28, 28], maxZoom: 16 });
  }, [map, rawGps, predictedRoute]);
  return null;
}

export default function RouteMap({ result }) {
  const rawGps = result?.raw_gps || [];
  const predicted = result?.predicted_route || [];
  const roadNetwork = result?.road_network || [];
  const center = rawGps[0] || predicted[0] || [0, 0];

  return (
    <div className="map-wrap">
      <MapContainer center={center} zoom={13} scrollWheelZoom className="route-map">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {roadNetwork.map((segment, i) => <Polyline key={`road-${i}`} positions={segment} pathOptions={{ color: '#64748b', weight: 3, opacity: 0.38 }} />)}
        {rawGps.length > 1 && <Polyline positions={rawGps} pathOptions={{ color: '#e5a23b', weight: 2, dashArray: '5 7', opacity: 0.8 }} />}
        {rawGps.map((point, i) => <CircleMarker key={`gps-${i}`} center={point} radius={3} pathOptions={{ color: '#e5a23b', fillColor: '#f4bd61', fillOpacity: 0.8, weight: 1 }} />)}
        {predicted.length > 1 && <Polyline positions={predicted} pathOptions={{ color: '#2dd4bf', weight: 5, opacity: 0.95 }} />}
        <FitBounds rawGps={rawGps} predictedRoute={predicted} />
      </MapContainer>
      <div className="map-legend">
        <span><i className="legend-dot gps"/> Raw GPS</span>
        <span><i className="legend-line route"/> Predicted route</span>
        <span><i className="legend-line roads"/> Road network</span>
      </div>
    </div>
  );
}
