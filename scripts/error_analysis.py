"""Where does the decoder still go wrong? Error taxonomy of the FIRST system (11 base features + tuned HMM).

Motivates the neighbour-point context features. Uses the 80 training trips only (never student-test or evaluator data).
  1. out-of-fold probabilities with the 11 base features (5-fold GroupKFold by trip, all rows),
  2. decode with the decoder settings tuned for that system (results/decoder_tuning/summary.json),
  3. classify every wrong point (true arc is among the candidates) as
       reverse  = predicted arc is the opposite-direction twin of the labelled arc
       adjacent = predicted arc is within 3 directed hops of the labelled arc (either direction)
       other    = anything else,
     and report how often the predicted arc is at least as close to the GPS point as the labelled arc.
Writes results/report/error_analysis.json.

  python scripts/error_analysis.py --train-dir "../Student folder/train" --train-csv work/train.csv.gz --model <11-feature model>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

import tune_decoder as T
from pathsnap.candidates import FEATURE_COLUMNS
from pathsnap.hmm.viterbi import HMMParams

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "artifacts", "candidate_model.joblib"),
                    help="only its hyper-parameters are reused; refit on the 11 base features")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "report", "error_analysis.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    base = joblib.load(args.model)
    df = T.read_table(args.train_csv)
    X, y = df[FEATURE_COLUMNS].to_numpy(np.float32), df["label"].to_numpy()
    p = np.zeros(len(df))
    for a, b in GroupKFold(5).split(X, y, df["trip_id"]):
        p[b] = clone(base).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    df["p"] = p
    cfg = json.load(open(os.path.join(ROOT, "results", "decoder_tuning", "summary.json")))["best_params"]
    params = HMMParams(**cfg)
    trips = T.prepare(df, p, args.train_dir, 0, set())
    dist = df.set_index(["trip_id", "point_idx", "arc_id"])["dist_m"]

    cat = dict(reverse=0, adjacent=0, other=0)
    not_farther, n_err, n_cov, n_nopred = 0, 0, 0, 0
    for tid, b in trips.items():
        pred = T.run_hmm(b, params).to_numpy()
        truth, g = b["truth"], b["graph"]
        cov = truth >= 0
        n_cov += int(cov.sum())
        for i in np.where(cov & (pred != truth))[0]:
            a, t = int(pred[i]), int(truth[i])
            n_err += 1
            if a < 0:
                cat["other"] += 1
                n_nopred += 1
                continue
            if g.from_node[a] == g.to_node[t] and g.to_node[a] == g.from_node[t]:
                cat["reverse"] += 1
            elif g.hops(a, t, 3) is not None or g.hops(t, a, 3) is not None:
                cat["adjacent"] += 1
            else:
                cat["other"] += 1
            try:
                not_farther += int(dist[(tid, i, a)] <= dist[(tid, i, t)] + 3.0)
            except KeyError:
                pass
    out = dict(partition="training (out-of-fold)", features=len(FEATURE_COLUMNS), covered_points=n_cov, errors=n_err,
               accuracy_pooled=1 - n_err / n_cov, counts=cat, shares={k: v / n_err for k, v in cat.items()},
               predicted_arc_not_farther_than_label_arc_share=not_farther / n_err, decoder_params=cfg)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
