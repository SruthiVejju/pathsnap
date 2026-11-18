"""Tune the HMM/Viterbi decoder parameters and compare classifier-only vs classifier+HMM.

Uses ONLY the training and student-test partitions (never evaluator data).

Inputs (all produced by existing scripts):
  --train-dir / --student-dir   raw trip folders (need .track/.arcs; the route is NOT read here)
  --train-csv / --student-csv   candidate tables from scripts/build_dataset.py (with labels)
  --model                       trained classifier (joblib, sklearn-style predict_proba)

Protocol
  * Training trips: classifier probabilities are OUT-OF-FOLD (GroupKFold by trip, the model is cloned and
    refit on the other folds). In-sample probabilities are overconfident and would mis-tune the decoder.
  * Student-test trips: probabilities from the final trained model (never fit on these trips).
  * Objective: mean over trips of pointwise accuracy of the HMM output on points whose reference arc is
    among the candidates ("covered"). Misses cannot be fixed by the decoder, so they are reported
    separately (coverage) and do not change the optimum.
  * Search: Bayesian optimisation (Optuna TPE, seed 42) on the TRAINING trips only. The student-test trips
    are then used to compare classifier-only / default HMM / tuned HMM.

Pointwise "truth" here is the monotonic route alignment from Part A (a derived label, not independent
ground truth): the reference arc of a point is the arc with label == 1 in the candidate table.

Example
  python scripts/tune_decoder.py --train-dir "../Student folder/train" --train-csv results/train.csv.gz \
      --student-dir "../Student folder/student_test" --student-csv results/student_test.csv.gz \
      --model artifacts/candidate_model.joblib --trials 40 --max-points 2000 --write-artifact
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap.candidates import FEATURE_COLUMNS
from pathsnap.data import load_trip
from pathsnap.hmm.graph import RoadGraph
from pathsnap.hmm.metrics import route_continuity
from pathsnap.hmm.viterbi import HMMParams, decode

SEED = 42


# ----------------------------------------------------------------------------- data preparation
def read_table(path):
    df = pd.read_csv(path, dtype={"trip_id": str})
    missing = [c for c in [*FEATURE_COLUMNS, "label", "point_idx", "arc_id"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}")
    return df


def oof_probabilities(model, df, folds, fit_rows):
    """Out-of-fold P(true arc) for every row, folds grouped by trip."""
    probs = np.zeros(len(df))
    X = df[FEATURE_COLUMNS].to_numpy(np.float32)
    y = df["label"].to_numpy()
    rng = np.random.default_rng(SEED)
    n_groups = df["trip_id"].nunique()
    for k, (tr, va) in enumerate(GroupKFold(n_splits=min(folds, n_groups)).split(X, y, df["trip_id"])):
        pick = tr if len(tr) <= fit_rows else rng.choice(tr, fit_rows, replace=False)
        m = clone(model).fit(X[pick], y[pick])
        probs[va] = m.predict_proba(X[va])[:, 1]
        print(f"  OOF fold {k + 1}: fit on {len(pick)} rows, scored {len(va)}", flush=True)
    return probs


def prepare(df, probs, trip_root, max_points, exclude):
    """Per-trip bundles holding everything decoding and scoring need (no route is read)."""
    trips = {}
    for tid, g in df.assign(p=probs).groupby("trip_id", sort=True):
        if tid in exclude:
            continue
        if max_points:
            g = g[g["point_idx"] < max_points]
        if g.empty:
            continue
        trip = load_trip(os.path.join(trip_root, tid), with_route=False)
        n = min(len(trip.track), max_points) if max_points else len(trip.track)
        truth = np.full(n, -1)
        hit = g[g["label"] == 1]
        truth[hit["point_idx"].to_numpy()] = hit["arc_id"].to_numpy()
        top = g.sort_values("p").groupby("point_idx").tail(1).set_index("point_idx")["arc_id"]
        trips[tid] = dict(
            graph=RoadGraph(trip.arcs), times=trip.track[:n, 2], n=n, truth=truth,
            cand=g[["point_idx", "arc_id"]].reset_index(drop=True), p=g["p"].to_numpy(),
            top1=top.reindex(range(n), fill_value=-1).to_numpy(),
        )
    return trips


# ----------------------------------------------------------------------------- scoring
def score_trip(b, pred):
    covered = b["truth"] >= 0
    if not covered.any():
        return None
    hit = pred[covered] == b["truth"][covered]
    return dict(acc_covered=float(hit.mean()), acc_all=float(hit.sum() / b["n"]),
                coverage=float(covered.mean()))


def run_hmm(b, params):
    return decode(b["cand"], b["p"], b["graph"], times=b["times"], params=params, n_points=b["n"])


def evaluate(trips, params):
    rows = []
    for tid, b in trips.items():
        pred = run_hmm(b, params)
        s = score_trip(b, pred.to_numpy())
        if s is None:
            continue
        rows.append(dict(trip_id=tid, continuity=route_continuity(pred, b["graph"]), **s))
    return pd.DataFrame(rows)


def evaluate_top1(trips):
    rows = []
    for tid, b in trips.items():
        s = score_trip(b, b["top1"])
        if s is None:
            continue
        cont = route_continuity(pd.Series(b["top1"]), b["graph"])
        rows.append(dict(trip_id=tid, continuity=cont, **s))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- search
def suggest(trial):
    # emission_weight is FIXED at 1.0: decoding only depends on the ratios between emission and transition
    # scores, so tuning all three would be redundant (the optimiser just drifts along a flat direction).
    return HMMParams(
        emission_weight=1.0,
        hop_penalty=trial.suggest_float("hop_penalty", 0.1, 100.0, log=True),
        unreachable_penalty=trial.suggest_float("unreachable_penalty", 1.0, 3000.0, log=True),
        max_hops=trial.suggest_int("max_hops", 2, 12),
        gap_s=trial.suggest_float("gap_s", 5.0, 300.0, log=True),
        uturn_penalty=trial.suggest_float("uturn_penalty", 1.0, 3000.0, log=True),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--student-dir", required=True)
    ap.add_argument("--student-csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="results/decoder_tuning")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fit-rows", type=int, default=60000, help="max rows to refit the classifier per OOF fold")
    ap.add_argument("--max-points", type=int, default=0, help="use only the first N points of each trip (speed); 0 = all")
    ap.add_argument("--max-trips", type=int, default=0, help="use only the first N training trips (speed); 0 = all")
    ap.add_argument("--exclude-trip", action="append", default=[], help="drop a trip from tuning/evaluation (repeatable)")
    ap.add_argument("--write-artifact", action="store_true", help="write the tuned params to artifacts/hmm_config.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    model = joblib.load(args.model)
    train_df, stud_df = read_table(args.train_csv), read_table(args.student_csv)
    if args.max_trips:
        keep = sorted(train_df["trip_id"].unique())[: args.max_trips]
        train_df = train_df[train_df["trip_id"].isin(keep)]
    print(f"Training table: {train_df['trip_id'].nunique()} trips, {len(train_df)} rows", flush=True)

    print("Out-of-fold probabilities for the training trips...", flush=True)
    train_p = oof_probabilities(model, train_df, args.folds, args.fit_rows)
    stud_p = model.predict_proba(stud_df[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]

    train = prepare(train_df, train_p, args.train_dir, args.max_points, set(args.exclude_trip))
    stud = prepare(stud_df, stud_p, args.student_dir, args.max_points, set(args.exclude_trip))
    print(f"Prepared {len(train)} training trips and {len(stud)} student-test trips", flush=True)

    default = HMMParams()
    base_train = evaluate(train, default)["acc_covered"].mean()
    print(f"Default decoder, training OOF accuracy (covered): {base_train:.4f}", flush=True)

    def objective(trial):
        return evaluate(train, suggest(trial))["acc_covered"].mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(asdict(default))        # trial 0 = the defaults, so "tuned >= default" is guaranteed
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = HMMParams(**{k: (int(v) if k == "max_hops" else float(v)) for k, v in study.best_params.items()})
    print(f"Best trial {study.best_trial.number}: {study.best_value:.4f}  params={asdict(best)}", flush=True)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    trials.to_csv(os.path.join(args.out_dir, "trials.csv"), index=False)

    # ---- comparison: classifier-only vs default HMM vs tuned HMM, on both partitions
    summary, per_trip = {}, []
    for name, trips in [("train_oof", train), ("student_test", stud)]:
        variants = {"classifier_only": evaluate_top1(trips),
                    "hmm_default": evaluate(trips, default),
                    "hmm_tuned": evaluate(trips, best)}
        summary[name] = {}
        for v, d in variants.items():
            summary[name][v] = {k: float(d[k].mean()) for k in ["acc_covered", "acc_all", "continuity", "coverage"]}
            summary[name][v]["n_trips"] = int(len(d))
            per_trip.append(d.assign(partition=name, variant=v))
    pd.concat(per_trip).to_csv(os.path.join(args.out_dir, "per_trip_comparison.csv"), index=False)

    record = dict(
        seed=SEED, trials=args.trials, folds=args.folds, max_points=args.max_points,
        excluded_trips=args.exclude_trip, objective="mean over trips of HMM pointwise accuracy on covered points "
        "(training trips, out-of-fold classifier probabilities)",
        search_space=dict(emission_weight="fixed 1.0 (redundant with the penalties)", hop_penalty="log 0.1-100",
                          unreachable_penalty="log 1-3000", max_hops="int 2-12", gap_s="log 5-300"),
        default_params=asdict(default), best_params=asdict(best), summary=summary,
        model=os.path.basename(args.model), seconds=round(time.time() - started, 1),
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(record, f, indent=2)
    if args.write_artifact:
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/hmm_config.json", "w") as f:
            json.dump(dict(params=asdict(best), tuned_on="train (out-of-fold)", trials=args.trials, seed=SEED), f, indent=2)
        print("Wrote artifacts/hmm_config.json", flush=True)

    print("\n=== accuracy on covered points / route continuity (mean over trips) ===")
    for part, d in summary.items():
        for v, m in d.items():
            print(f"{part:13s} {v:16s} acc={m['acc_covered']:.4f}  cont={m['continuity']:.4f}  (n={m['n_trips']})")
    print(f"done in {record['seconds']} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
