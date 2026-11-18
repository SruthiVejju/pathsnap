"""Build candidate tables for a partition and optionally audit excluded trips."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

from pathsnap.candidates import MAX_CAND, RADIUS, add_labels, build_candidates
from pathsnap.data import list_trips, load_trip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition_dir")
    parser.add_argument("output_csv")
    parser.add_argument("--max-trips", type=int)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--radius", type=float, default=RADIUS)
    parser.add_argument("--max-cand", type=int, default=MAX_CAND)
    parser.add_argument(
        "--exclude-trip",
        action="append",
        default=[],
        metavar="TRIP_ID",
        help="Exclude a trip from the candidate table but retain its audit metrics.",
    )
    parser.add_argument(
        "--exclude-reason",
        action="append",
        default=[],
        metavar="TRIP_ID=REASON",
        help="Record a reason for excluding a trip (repeatable).",
    )
    args = parser.parse_args()

    trip_dirs = list_trips(args.partition_dir)
    if args.max_trips is not None:
        trip_dirs = trip_dirs[:args.max_trips]
    if not trip_dirs:
        parser.error(f"No trip directories found in {args.partition_dir}")

    output_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(output_dir, exist_ok=True)
    frames, miss, excluded = [], [], []
    exclusion_reasons = {}
    for item in args.exclude_reason:
        trip_id, separator, reason = item.partition("=")
        if not separator or not trip_id or not reason:
            parser.error("--exclude-reason must be formatted as TRIP_ID=REASON")
        exclusion_reasons[trip_id] = reason

    for trip_dir in trip_dirs:
        started = time.time()
        trip = load_trip(trip_dir, with_route=not args.no_labels)
        candidates = build_candidates(
            trip, radius=args.radius, max_cand=args.max_cand
        )
        is_excluded = trip.trip_id in args.exclude_trip

        if not args.no_labels:
            candidates, _, true_distances = add_labels(candidates, trip)
            has_label = (
                candidates.groupby("point_idx")["label"]
                .max()
                .reindex(range(len(trip.pxy)))
                .fillna(0)
            )
            metrics = {
                "trip_id": trip.trip_id,
                "n_points": len(trip.pxy),
                "miss_rate": 1 - has_label.mean(),
                "mean_true_dist": true_distances.mean(),
                "mean_cands": candidates.groupby("point_idx").size().mean(),
                "included_in_output": not is_excluded,
            }
            miss.append(metrics)
            print(
                f"{trip.trip_id}: {len(candidates)} candidate rows, "
                f"miss {metrics['miss_rate']:.3f}"
                f"{', excluded from table' if is_excluded else ''}, "
                f"{time.time() - started:.1f}s",
                flush=True,
            )
        else:
            print(
                f"{trip.trip_id}: {len(candidates)} candidate rows"
                f"{', excluded from table' if is_excluded else ''}",
                flush=True,
            )

        if is_excluded:
            excluded.append(
                {
                    "trip_id": trip.trip_id,
                    "reason": exclusion_reasons.get(
                        trip.trip_id,
                        "Excluded from this output by --exclude-trip; see miss report.",
                    ),
                }
            )
        else:
            frames.append(candidates)

    if not frames:
        parser.error("No trips remain for the candidate table.")
    pd.concat(frames, ignore_index=True).to_csv(args.output_csv, index=False)

    base, _ = os.path.splitext(args.output_csv)
    if args.output_csv.endswith(".csv.gz"):
        base = args.output_csv[:-7]
    if miss:
        miss_path = f"{base}_miss.csv"
        miss_frame = pd.DataFrame(miss)
        miss_frame.to_csv(miss_path, index=False)
        included = miss_frame[miss_frame["included_in_output"]]
        print(
            f"Included trips: {len(included)}; mean miss rate: "
            f"{included['miss_rate'].mean():.4f}; maximum: "
            f"{included['miss_rate'].max():.4f}"
        )
        if excluded:
            pd.DataFrame(excluded).to_csv(f"{base}_excluded.csv", index=False)


if __name__ == "__main__":
    main()
