
"""Generate an interactive Folium GPS trajectory map.

Only permitted development trips are accepted.
Coordinate convention: input [longitude, latitude].
Folium expects [latitude, longitude].

Never load evaluator ground-truth routes into this map.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from html import escape

import folium
import pandas as pd


def read_track(path: Path) -> pd.DataFrame:
    """Read and validate GPS observations."""

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    if frame.empty:
        raise ValueError(f"No GPS observations found in {path}")

    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()

    valid = (
        frame["longitude"].between(-180, 180)
        & frame["latitude"].between(-90, 90)
    )

    frame = (
        frame.loc[valid]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValueError("No valid GPS coordinates remain.")

    return frame


def add_legend(m: folium.Map) -> None:
    """Add a custom map legend."""

    legend_html = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        z-index: 9999;
        background: white;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #cccccc;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        font-family: Arial;
        font-size: 13px;
    ">
        <b style="font-size:14px;">PathSnap - Map Legend</b>
        <hr style="margin:7px 0;">

        <div>
            <span style="color:#2563eb;">●</span>
            Raw GPS observations
        </div>

        <div>
            <span style="color:#16a34a;">●</span>
            Starting location
        </div>

        <div>
            <span style="color:#dc2626;">●</span>
            Ending location
        </div>

        <div>
            <span style="color:#9333ea;">━</span>
            Reconstructed route (future overlay)
        </div>
    </div>
    """

    from branca.element import Element

    m.get_root().html.add_child(Element(legend_html))


def add_trip_information(
    m: folium.Map,
    trip_id: str,
    gps: pd.DataFrame
) -> None:
    """Display basic trip information."""

    first = gps.iloc[0]
    last = gps.iloc[-1]

    info_html = f"""
    <div style="
        position: fixed;
        top: 15px;
        left: 60px;
        z-index: 9999;
        background: white;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #cccccc;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        font-family: Arial;
        font-size: 12px;
    ">
        <b style="font-size:15px;">PathSnap | Trip {escape(trip_id)}</b>
        <hr style="margin:6px 0;">
        GPS observations: <b>{len(gps):,}</b><br>
        Start: {first['latitude']:.5f}, {first['longitude']:.5f}<br>
        End: {last['latitude']:.5f}, {last['longitude']:.5f}
    </div>
    """

    from branca.element import Element

    m.get_root().html.add_child(Element(info_html))


def generate_map(track_path: Path, output_path: Path) -> None:
    """Generate the interactive GPS map."""

    gps = read_track(track_path)

    points = gps[
        ["latitude", "longitude"]
    ].values.tolist()

    center = [
        float(gps["latitude"].mean()),
        float(gps["longitude"].mean())
    ]

    trip_id = track_path.stem

    # Initialize map
    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles=None
    )

    # Esri basemap - working tile provider
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Sources: Esri, HERE, Garmin, USGS, "
            "© OpenStreetMap contributors"
        ),
        name="Esri World Street Map",
        overlay=False,
        control=True
    ).add_to(m)

    # Raw GPS observations layer
    raw_layer = folium.FeatureGroup(
        name="Raw GPS observations",
        show=True
    )

    # GPS trajectory
    folium.PolyLine(
        points,
        color="#2563eb",
        weight=4,
        opacity=0.85,
        tooltip="Raw GPS trajectory"
    ).add_to(raw_layer)

    # Individual GPS points
    for i, point in enumerate(points):

        folium.CircleMarker(
            location=point,
            radius=2,
            color="#1d4ed8",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.8,
            weight=1,
            tooltip=f"GPS observation {i + 1}"
        ).add_to(raw_layer)

    raw_layer.add_to(m)

    # Starting location
    start = points[0]

    folium.Marker(
        location=start,
        tooltip="Starting location",
        popup=(
            f"Start of trip {escape(trip_id)}"
            f"<br>Latitude: {start[0]:.6f}"
            f"<br>Longitude: {start[1]:.6f}"
        ),
        icon=folium.Icon(
            color="green",
            icon="play",
            prefix="fa"
        )
    ).add_to(m)

    # Ending location
    end = points[-1]

    folium.Marker(
        location=end,
        tooltip="Ending location",
        popup=(
            f"End of trip {escape(trip_id)}"
            f"<br>Latitude: {end[0]:.6f}"
            f"<br>Longitude: {end[1]:.6f}"
        ),
        icon=folium.Icon(
            color="red",
            icon="stop",
            prefix="fa"
        )
    ).add_to(m)

    # Information and legend
    add_trip_information(m, trip_id, gps)
    add_legend(m)

    # Layer controls
    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # Automatically fit trajectory
    m.fit_bounds(points)

    # Save map
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    m.save(str(output_path))

    print(f"Trip ID: {trip_id}")
    print(f"GPS observations: {len(gps)}")
    print(f"Map saved to: {output_path.resolve()}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--track",
        required=True,
        type=Path,
        help="Path to a permitted .track file"
    )

    parser.add_argument(
        "--output",
        default=Path(
            "maps/output/raw_gps_preview.html"
        ),
        type=Path
    )

    args = parser.parse_args()

    generate_map(
        args.track,
        args.output
    )


if __name__ == "__main__":
    main()
