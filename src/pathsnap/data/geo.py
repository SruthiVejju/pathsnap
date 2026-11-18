"""Per-trip metric projection for trips spanning multiple UTM zones."""
import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_projector(lonlat_ref: np.ndarray) -> Transformer:
    """Choose a UTM zone from the mean coordinates of a trip."""
    lon, lat = float(np.mean(lonlat_ref[:, 0])), float(np.mean(lonlat_ref[:, 1]))
    return Transformer.from_crs(4326, utm_epsg(lon, lat), always_xy=True)


def project(transformer: Transformer, lonlat: np.ndarray) -> np.ndarray:
    x, y = transformer.transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack((x, y))
