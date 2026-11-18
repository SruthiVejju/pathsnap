# Decoder tuning record

`summary.json`, `trials.csv`, `per_trip_comparison.csv` are the FINAL run (run 3, 60 Bayesian/TPE trials, seed 42).
`earlier_runs/` keeps the two earlier runs so the search history is complete:

| Run | Search | Best train OOF acc | Student-test acc | Why replaced |
|---|---|---|---|---|
| 1 | 30 trials, narrow ranges, emission_weight tuned | 0.9359 | 0.9516 | best trial was the last one and two values sat at range edges |
| 2 | 60 trials, wider ranges, emission_weight tuned | 0.9511 | 0.9538 | emission_weight hit its lower bound again: it is redundant with the penalties (only their ratios matter) |
| 3 (final) | 60 trials, emission_weight fixed at 1.0 | 0.9511 | 0.9547 | identifiable search; best hop_penalty 14.06 matches run 2's effective ratio |

Final settings are in `artifacts/hmm_config.json`. Selection used the TRAINING objective only; student-test is a comparison.
Accuracy = pointwise accuracy on points whose reference arc is among the candidates, against Part A's monotonic-alignment
labels (derived labels, not independent ground truth). Only the first 1500 points of each trip were used for speed.
Classifier probabilities for training trips are out-of-fold (GroupKFold by trip, 5 folds).

Reproduce: `python scripts/tune_decoder.py --train-dir ... --train-csv ... --student-dir ... --student-csv ...
--model artifacts/candidate_model.joblib --trials 60 --max-points 1500 --write-artifact`
(tables from `scripts/build_dataset.py`).
