"""Create reproducible, data-derived EDA figures from the training partition.

Expected dataset layout:
  DATASET_ROOT/train/<trip_id>/<trip_id>.track

The script deliberately does not read .route files. Evaluator-test data must not
be passed to this script or used for EDA/model selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_track(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["longitude", "latitude", "timestamp"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df[df.longitude.between(-180, 180) & df.latitude.between(-90, 90)]
    return df.sort_values("timestamp").reset_index(drop=True)


def collect_training_stats(dataset_root: Path) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    train_dir = dataset_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected training directory: {train_dir}")
    summaries: list[dict] = []
    tracks: list[tuple[str, pd.DataFrame]] = []
    for track_path in sorted(train_dir.glob("*/*.track")):
        trip_id = track_path.parent.name
        df = load_track(track_path)
        if df.empty:
            summaries.append({"trip_id": trip_id, "observations": 0, "median_gap_s": None, "max_gap_s": None})
            continue
        gaps = df.timestamp.diff().dropna()
        summaries.append({
            "trip_id": trip_id,
            "observations": len(df),
            "median_gap_s": float(gaps.median()) if not gaps.empty else None,
            "max_gap_s": float(gaps.max()) if not gaps.empty else None,
            "invalid_rows_removed": None,
        })
        tracks.append((trip_id, df))
    if not summaries:
        raise FileNotFoundError(f"No .track files found under {train_dir}")
    return pd.DataFrame(summaries), tracks


def make_figures(summary: pd.DataFrame, tracks: list[tuple[str, pd.DataFrame]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(summary.loc[summary.observations > 0, "observations"], bins=20, edgecolor="black")
    ax.set(title="GPS observations per training trip", xlabel="Number of valid GPS observations", ylabel="Number of trips")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "training_observations_per_trip.png", dpi=180); plt.close(fig)

    gaps = [df.timestamp.diff().dropna() for _, df in tracks]
    all_gaps = pd.concat(gaps, ignore_index=True) if gaps else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not all_gaps.empty:
        ax.hist(all_gaps.clip(upper=all_gaps.quantile(0.99)), bins=35, edgecolor="black")
    ax.set(title="GPS timestamp intervals (training trips; upper 1% clipped for display)", xlabel="Interval between observations (seconds)", ylabel="Number of intervals")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "training_timestamp_intervals.png", dpi=180); plt.close(fig)

    if tracks:
        trip_id, df = tracks[0]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(df.longitude, df.latitude, s=5, alpha=0.65)
        ax.set(title=f"Raw GPS trajectory – training trip {trip_id}", xlabel="Longitude", ylabel="Latitude")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(output_dir / f"training_trip_{trip_id}_trajectory.png", dpi=180); plt.close(fig)

    summary.to_csv(output_dir / "training_trip_eda_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Directory containing train/ and student_test/ subdirectories")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/output/eda"))
    args = parser.parse_args()
    summary, tracks = collect_training_stats(args.dataset_root)
    make_figures(summary, tracks, args.output_dir)
    print(f"Processed {len(summary)} training trips; figures saved to {args.output_dir.resolve()}")
    print("Only .track files under train/ were read; no route labels were loaded.")


if __name__ == "__main__":
    main()
