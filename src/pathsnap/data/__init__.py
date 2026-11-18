"""Loading raw PathSnap trips and projecting each trip to metres."""
import os
from dataclasses import dataclass

import numpy as np

from .geo import make_projector, project


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray = None
    pxy: np.ndarray = None
    nxy: np.ndarray = None


def load_trip(trip_dir, with_route=True):
    trip_id = os.path.basename(os.path.normpath(trip_dir))

    def path_for(extension):
        return os.path.join(trip_dir, f"{trip_id}.{extension}")

    track = np.atleast_2d(np.loadtxt(path_for("track")))
    nodes = np.atleast_2d(np.loadtxt(path_for("nodes")))
    arcs = np.atleast_2d(np.loadtxt(path_for("arcs"), dtype=int))
    route = None
    if with_route and os.path.exists(path_for("route")):
        route = np.atleast_1d(np.loadtxt(path_for("route"), dtype=int))

    transformer = make_projector(track[:, :2])
    pxy = project(transformer, track[:, :2])
    nxy = project(transformer, nodes[:, :2])
    return Trip(trip_id, track, nodes, arcs, route, pxy, nxy)


def list_trips(partition_dir):
    return sorted(
        os.path.join(partition_dir, directory)
        for directory in os.listdir(partition_dir)
        if os.path.isdir(os.path.join(partition_dir, directory))
    )
