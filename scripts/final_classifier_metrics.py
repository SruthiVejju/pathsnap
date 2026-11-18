"""Classifier-level metrics of the FINAL model (23 features) for the technical report.

Writes results/report/final_classifier_metrics.json with
  * 5-fold GroupKFold (by trip) cross-validation on the 80 training trips: macro-F1 per fold, log loss
  * student-test: macro precision/recall/F1, observation-level top-1 accuracy, log loss, confusion matrix,
    inference time (predict_proba over all candidate rows)
Uses only the training and student-test partitions (never evaluator data). The model is the one in
artifacts/ (hyper-parameters reused, refit per fold exactly as in scripts/train_context_model.py).

Example
  python scripts/final_classifier_metrics.py --train-dir "../Student folder/train" --train-csv work/train.csv.gz \
      --student-dir "../Student folder/student_test" --student-csv work/student_test.csv.gz
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from train_context_model import with_context
from pathsnap.context import EXTENDED_COLUMNS

ROOT = os.path.join(os.path.dirname(__file__), "..")


def top1_accuracy(df, p):
    """Observation-level top-1: per GPS point take the highest-scoring candidate, compare with the label.
    Points whose true arc is not among the candidates count as misses (denominator = all points)."""
    d = df[["trip_id", "point_idx", "label"]].assign(p=p)
    best = d.sort_values("p").groupby(["trip_id", "point_idx"]).tail(1)
    hits = int(best["label"].sum())
    n_points = d.groupby(["trip_id", "point_idx"]).ngroups
    return hits / n_points, n_points


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--student-dir", required=True)
    ap.add_argument("--student-csv", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts", "candidate_model.joblib"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "report", "final_classifier_metrics.json"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = joblib.load(args.model)
    train, stud = with_context(args.train_csv, args.train_dir), with_context(args.student_csv, args.student_dir)
    Xt, yt = train[EXTENDED_COLUMNS].to_numpy(np.float32), train["label"].to_numpy()
    Xs, ys = stud[EXTENDED_COLUMNS].to_numpy(np.float32), stud["label"].to_numpy()

    # ---- cross-validation on the training trips
    oof, fold_f1, fold_ll = np.zeros(len(train)), [], []
    for a, b in GroupKFold(n_splits=args.folds).split(Xt, yt, train["trip_id"]):
        m = clone(model).fit(Xt[a], yt[a])
        oof[b] = m.predict_proba(Xt[b])[:, 1]
        fold_f1.append(float(f1_score(yt[b], oof[b] > 0.5, average="macro")))
        fold_ll.append(float(log_loss(yt[b], oof[b], labels=[0, 1])))
        print(f"fold: macro-F1 {fold_f1[-1]:.4f}", flush=True)
    cv_top1, _ = top1_accuracy(train, oof)

    # ---- student-test with the model fitted on all training rows (the shipped model)
    t0 = time.perf_counter()
    p = model.predict_proba(Xs)[:, 1]
    inference = time.perf_counter() - t0
    pr, rc, f1, _ = precision_recall_fscore_support(ys, p > 0.5, average="macro", zero_division=0)
    top1, n_points = top1_accuracy(stud, p)
    train_p = model.predict_proba(Xt)[:, 1]
    out = dict(
        features=len(EXTENDED_COLUMNS), folds=args.folds,
        cv=dict(macro_f1=float(np.mean(fold_f1)), fold_macro_f1=fold_f1, log_loss=float(log_loss(yt, oof, labels=[0, 1])),
                top1_accuracy=cv_top1, macro_f1_pooled=float(f1_score(yt, oof > 0.5, average="macro"))),
        train_in_sample=dict(macro_f1=float(f1_score(yt, train_p > 0.5, average="macro")),
                             log_loss=float(log_loss(yt, train_p, labels=[0, 1]))),
        student_test=dict(macro_f1=float(f1), macro_precision=float(pr), macro_recall=float(rc),
                          top1_accuracy=top1, log_loss=float(log_loss(ys, p, labels=[0, 1])),
                          candidate_rows=int(len(ys)), gps_points=int(n_points), inference_seconds=inference,
                          confusion_matrix=confusion_matrix(ys, p > 0.5, labels=[0, 1]).tolist()),
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("cv", "student_test")}, indent=1)[:900])


if __name__ == "__main__":
    main()
