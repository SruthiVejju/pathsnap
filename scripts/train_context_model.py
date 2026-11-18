"""Train the final classifier on base + neighbour-point context features and tune the HMM decoder for it.

Uses ONLY the training and student-test partitions (never evaluator data).

Steps (seed 42 throughout)
  1. Read the candidate tables from scripts/build_dataset.py and add the context features of
     pathsnap.context (route is never read here; labels come from the table).
  2. Out-of-fold probabilities on the 80 training trips (GroupKFold by trip, model cloned from the
     existing tuned classifier's hyper-parameters, fitted on ALL rows of the other folds).
  3. Bayesian decoder tuning (Optuna TPE) on those out-of-fold probabilities, objective = mean pointwise
     accuracy on covered points (same protocol as scripts/tune_decoder.py).
  4. Final classifier fitted on all training rows; evaluation on the student-test trips for
     classifier-only / default decoder / tuned decoder.
  5. With --write-artifact: artifacts/candidate_model.joblib, model_config.json, hmm_config.json.

Example
  python scripts/train_context_model.py --train-dir "../Student folder/train" --train-csv work/train.csv.gz \
      --student-dir "../Student folder/student_test" --student-csv work/student_test.csv.gz \
      --base-model artifacts/candidate_model.joblib --trials 60 --write-artifact
"""
import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

import tune_decoder as T
from pathsnap.candidates import MAX_CAND, RADIUS
from pathsnap.context import CONTEXT_COLUMNS, EXTENDED_COLUMNS, add_context
from pathsnap.data import load_trip
from pathsnap.hmm.viterbi import HMMParams

SEED = T.SEED


def with_context(csv_path, trip_root):
    df = pd.read_csv(csv_path, dtype={"trip_id": str})
    parts = [add_context(g, load_trip(os.path.join(trip_root, tid), with_route=False))
             for tid, g in df.groupby("trip_id", sort=True)]
    return pd.concat(parts, ignore_index=True)


def oof(model, df, folds):
    X, y = df[EXTENDED_COLUMNS].to_numpy(np.float32), df["label"].to_numpy()
    p = np.zeros(len(df))
    for k, (a, b) in enumerate(GroupKFold(n_splits=folds).split(X, y, df["trip_id"])):
        p[b] = clone(model).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        print(f"  OOF fold {k + 1}/{folds}: fit on {len(a)} rows, scored {len(b)}", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--student-dir", required=True)
    ap.add_argument("--student-csv", required=True)
    ap.add_argument("--base-model", required=True, help="existing classifier; only its hyper-parameters are reused")
    ap.add_argument("--out-dir", default="results/context_model")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--write-artifact", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    base = joblib.load(args.base_model)
    train, stud = with_context(args.train_csv, args.train_dir), with_context(args.student_csv, args.student_dir)
    print(f"train {train['trip_id'].nunique()} trips / {len(train)} rows; student {stud['trip_id'].nunique()} trips", flush=True)

    t0 = time.time()
    train["p"] = oof(base, train, args.folds)
    oof_seconds = time.time() - t0
    print(f"OOF log-loss {log_loss(train.label, train.p):.4f}, macro-F1 {f1_score(train.label, train.p > 0.5, average='macro'):.4f}")

    trips_tr = T.prepare(train, train["p"].to_numpy(), args.train_dir, 0, set())
    default = HMMParams()

    def suggest(trial):
        h = T.suggest(trial)
        return h

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({k: v for k, v in asdict(default).items() if k not in ("emission_weight", "uturn_penalty")}
                        | {"uturn_penalty": default.hop_penalty})
    study.optimize(lambda t: T.evaluate(trips_tr, suggest(t))["acc_covered"].mean(), n_trials=args.trials)
    best = HMMParams(**{k: (int(v) if k == "max_hops" else float(v)) for k, v in study.best_params.items()})
    print(f"best trial {study.best_trial.number}: {study.best_value:.4f} {asdict(best)}", flush=True)
    study.trials_dataframe(attrs=("number", "value", "params", "duration")).to_csv(
        os.path.join(args.out_dir, "trials.csv"), index=False)

    t0 = time.time()
    final = clone(base).fit(train[EXTENDED_COLUMNS].to_numpy(np.float32), train["label"].to_numpy())
    fit_seconds = time.time() - t0
    stud["p"] = final.predict_proba(stud[EXTENDED_COLUMNS].to_numpy(np.float32))[:, 1]
    trips_st = T.prepare(stud, stud["p"].to_numpy(), args.student_dir, 0, set())

    summary, per_trip = {}, []
    for name, trips in [("train_oof", trips_tr), ("student_test", trips_st)]:
        variants = {"classifier_only": T.evaluate_top1(trips), "hmm_default": T.evaluate(trips, default),
                    "hmm_tuned": T.evaluate(trips, best)}
        summary[name] = {}
        for v, d in variants.items():
            summary[name][v] = {k: float(d[k].mean()) for k in ["acc_covered", "acc_all", "continuity", "coverage"]}
            summary[name][v]["n_trips"] = int(len(d))
            per_trip.append(d.assign(partition=name, variant=v))
    pd.concat(per_trip).to_csv(os.path.join(args.out_dir, "per_trip_comparison.csv"), index=False)

    record = dict(seed=SEED, folds=args.folds, trials=args.trials, features=EXTENDED_COLUMNS, context_features=CONTEXT_COLUMNS,
                  base_model_params=base.get_params(), default_params=asdict(default), best_params=asdict(best),
                  summary=summary, oof_seconds=round(oof_seconds, 1), final_fit_seconds=round(fit_seconds, 1),
                  total_seconds=round(time.time() - started, 1))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)

    if args.write_artifact:
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(final, "artifacts/candidate_model.joblib")
        with open("artifacts/model_config.json", "w") as f:
            json.dump(dict(seed=SEED, radius_m=RADIUS, max_candidates=MAX_CAND, feature_columns=EXTENDED_COLUMNS,
                           label_alignment="monotonic ordered-route dynamic programming",
                           model="HistGradientBoostingClassifier (hyper-parameters from the tuned Part B model)",
                           fit_rows="all training rows (no cap)", train_trips=int(train["trip_id"].nunique()),
                           python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                           scikit_learn=sklearn.__version__), f, indent=2)
        with open("artifacts/hmm_config.json", "w") as f:
            json.dump(dict(params=asdict(best), tuned_on="train (out-of-fold, context-feature model)",
                           trials=args.trials, seed=SEED), f, indent=2)
        print("wrote artifacts/candidate_model.joblib, model_config.json, hmm_config.json")

    print("\n=== accuracy on covered points / route continuity (mean over trips) ===")
    for part, d in summary.items():
        for v, m in d.items():
            print(f"{part:13s} {v:16s} acc={m['acc_covered']:.4f}  cont={m['continuity']:.4f}  (n={m['n_trips']})")
    print(f"done in {record['total_seconds']} s -> {args.out_dir}")


if __name__ == "__main__":
    main()
