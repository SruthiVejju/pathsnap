"""Interactive Folium map. usage: python scripts/folium_map.py <trip_dir> <out.html> [candidates.csv.gz]
Layers: GPS points, reference route, (optional) chosen arcs from a prediction CSV with columns point_idx,arc_id."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import folium
import pandas as pd
from pathsnap.data import load_trip

trip = load_trip(sys.argv[1])
m = folium.Map(location=[trip.track[:, 1].mean(), trip.track[:, 0].mean()], zoom_start=15, tiles="OpenStreetMap")
gps = folium.FeatureGroup("Raw GPS points (blue)")
for lon, lat, t in trip.track[::3]:
    folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, popup=f"t={t:.0f}s").add_to(gps)
gps.add_to(m)
ref = folium.FeatureGroup("Reference route (green)")
for a in trip.route:
    f, t = trip.arcs[a]
    folium.PolyLine([trip.nodes[f][::-1], trip.nodes[t][::-1]], color="green", weight=5).add_to(ref)
ref.add_to(m)
if len(sys.argv) > 3:
    pr = pd.read_csv(sys.argv[3])
    pred = folium.FeatureGroup("Predicted route (red)")
    for a in pr["arc_id"].unique():
        f, t = trip.arcs[a]
        folium.PolyLine([trip.nodes[f][::-1], trip.nodes[t][::-1]], color="red", weight=3, dash_array="6").add_to(pred)
    pred.add_to(m)
folium.LayerControl().add_to(m)
m.save(sys.argv[2]); print("saved", sys.argv[2])
