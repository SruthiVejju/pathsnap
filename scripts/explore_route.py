"""Quick structural check of how .route, .arcs, .nodes and .track relate.

Questions answered per trip (partition given on the command line):
  1. Are consecutive route arcs connected (to_node of arc i == from_node of arc i+1)?
  2. How far (metres) is each GPS point from the nearest arc of ITS OWN route?
  3. Basic size / time-gap stats.

Usage: python scripts/explore_route.py train|student_test
"""
import sys

import numpy as np
import pandas as pd

from pathsnap.config import CFG, data_root, trip_ids
from pathsnap.data.geo import make_projector, project


def load_trip(partition: str, tid: str):
    d = data_root() / CFG["train_dir" if partition == "train" else "student_test_dir"] / tid
    track = np.loadtxt(d / f"{tid}.track")
    nodes = np.loadtxt(d / f"{tid}.nodes")
    arcs = np.loadtxt(d / f"{tid}.arcs", dtype=int)
    route = np.loadtxt(d / f"{tid}.route", dtype=int, ndmin=1)
    return track, nodes, arcs, route


def point_seg_dist(P, A, B):
    """Distance from each point in P (n,2) to each segment A[j]-B[j] (m,2) -> (n,m)."""
    AB = B - A
    L2 = (AB ** 2).sum(1)
    L2[L2 == 0] = 1e-12
    AP = P[:, None, :] - A[None, :, :]
    t = np.clip((AP * AB[None]).sum(2) / L2[None], 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(P[:, None, :] - proj, axis=2)


def main(partition: str):
    rows = []
    for tid in trip_ids(partition):
        track, nodes, arcs, route = load_trip(partition, tid)
        tf = make_projector(track[:, :2])
        N = project(tf, nodes)
        P = project(tf, track[:, :2])

        frm, to = arcs[route, 0], arcs[route, 1]
        connected = (to[:-1] == frm[1:]).mean() if len(route) > 1 else np.nan
        d = point_seg_dist(P, N[frm], N[to])
        dmin = d.min(1)
        dt = np.diff(track[:, 2])
        rows.append(dict(
            trip=tid, n_pts=len(track), n_nodes=len(nodes), n_arcs=len(arcs), n_route=len(route),
            route_connected=connected, dist_med=np.median(dmin), dist_p95=np.percentile(dmin, 95),
            dist_max=dmin.max(), dt_ne1=(dt != 1).mean(), dt_max=dt.max(),
            nan_pts=int(np.isnan(track).sum()), dup_route=len(route) - len(set(route)),
            route_len_m=np.linalg.norm(N[to] - N[frm], axis=1).sum(),
        ))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.describe().T[["mean", "min", "50%", "max"]].round(3))
    print("\nTrips with disconnected routes:\n", df[df.route_connected < 1][["trip", "route_connected"]])
    print("\nWorst GPS-to-route distances:\n", df.nlargest(5, "dist_max")[["trip", "dist_med", "dist_p95", "dist_max"]])
    print("\nLargest time gaps:\n", df.nlargest(5, "dt_max")[["trip", "dt_max", "dt_ne1"]])
    df.to_csv(f"results/explore_route_{partition}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
