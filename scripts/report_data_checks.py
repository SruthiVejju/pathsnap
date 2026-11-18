"""Raw-data quality checks for the technical report (missing values, timestamps, graph sizes) per partition.

Covers the training and student-test partitions only (never the evaluator partition).
Writes results/report/data_checks.csv (one row per partition) and results/report/data_checks_per_trip.csv.

  python scripts/report_data_checks.py "../Student folder"
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.data import list_trips

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(d, ext, dtype=float):
    t = os.path.basename(d)
    return np.atleast_2d(np.loadtxt(os.path.join(d, f"{t}.{ext}"), dtype=dtype))


def main(data_root):
    rows = []
    for part in ("train", "student_test"):
        for d in list_trips(os.path.join(data_root, part)):
            track, nodes, arcs = load(d, "track"), load(d, "nodes"), load(d, "arcs", int)
            route = np.atleast_1d(np.loadtxt(os.path.join(d, f"{os.path.basename(d)}.route"), dtype=int))
            dt = np.diff(track[:, 2])
            rows.append(dict(
                partition=part, trip_id=os.path.basename(d), gps_points=len(track), nodes=len(nodes), arcs=len(arcs),
                route_arcs=len(route),
                nan_or_inf_track=int((~np.isfinite(track)).sum()), nan_or_inf_nodes=int((~np.isfinite(nodes)).sum()),
                nonmonotonic_timestamps=int((dt < 0).sum()), duplicate_timestamps=int((dt == 0).sum()),
                median_dt_s=float(np.median(dt)), max_dt_s=float(dt.max()), gaps_over_30s=int((dt > 30).sum()),
                arcs_bad_node_ref=int(((arcs < 0) | (arcs >= len(nodes))).any(axis=1).sum()),
                route_bad_arc_ref=int(((route < 0) | (route >= len(arcs))).sum()),
                lon_min=float(track[:, 0].min()), lon_max=float(track[:, 0].max()),
            ))
    per = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "report")
    os.makedirs(out, exist_ok=True)
    per.to_csv(os.path.join(out, "data_checks_per_trip.csv"), index=False)
    agg = per.groupby("partition").agg(
        trips=("trip_id", "count"), gps_points=("gps_points", "sum"), median_points_per_trip=("gps_points", "median"),
        nodes_median=("nodes", "median"), arcs_median=("arcs", "median"), route_arcs_median=("route_arcs", "median"),
        nan_or_inf_values=("nan_or_inf_track", "sum"), nan_or_inf_nodes=("nan_or_inf_nodes", "sum"),
        nonmonotonic_timestamps=("nonmonotonic_timestamps", "sum"), duplicate_timestamps=("duplicate_timestamps", "sum"),
        median_dt_s=("median_dt_s", "median"), max_dt_s=("max_dt_s", "max"), gaps_over_30s=("gaps_over_30s", "sum"),
        trips_with_gap_over_30s=("gaps_over_30s", lambda s: int((s > 0).sum())),
        arcs_bad_node_ref=("arcs_bad_node_ref", "sum"), route_bad_arc_ref=("route_bad_arc_ref", "sum"))
    agg.to_csv(os.path.join(out, "data_checks.csv"))
    print(agg.T.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "Student folder"))
