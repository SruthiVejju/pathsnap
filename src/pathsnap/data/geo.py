"""Per-trip metric projection.

Trips come from all over the world (lon -100..137, 23 UTM zones in training data),
so a single fixed CRS is wrong. Each trip is projected to the UTM zone of its own
GPS centroid; distances are then in metres with <0.1% scale error.
"""
import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_projector(lon_lat_ref: np.ndarray) -> Transformer:
    """lon_lat_ref: (n,2) array, typically the trip's GPS track; used only to pick the zone."""
    lon, lat = float(np.mean(lon_lat_ref[:, 0])), float(np.mean(lon_lat_ref[:, 1]))
    return Transformer.from_crs(4326, utm_epsg(lon, lat), always_xy=True)


def project(transformer: Transformer, lonlat: np.ndarray) -> np.ndarray:
    x, y = transformer.transform(lonlat[:, 0], lonlat[:, 1])
    return np.c_[x, y]
