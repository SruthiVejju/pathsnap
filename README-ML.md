# PathSnap model experiments

Run from the directory containing `training/` with Python 3.12.

```sh
python -m pip install -r requirements-ml.txt
python -m training.experiments "/path/to/Student folder.zip"
python -m training.plots
```

The default run compares logistic regression, Gaussian naive Bayes, a decision tree, Extra Trees and histogram gradient boosting. All five use the same features and three trip-grouped folds from the 80 training trips. Each fit uses a deterministic sample of up to 60,000 candidate rows drawn from its training trips. Validation and student-test scoring use every available observation. The 10 student-test trips are scored separately. The evaluator partition is outside this experiment.

## Labels and evaluation

The reference file lists a sequence of arcs for the entire trip. For each GPS point, the experiment uses the geometrically closest arc on that reference route as an approximate point label. This method can mislabel points near parallel roads, crossings and revisited segments. It shares the distance signal with the candidate features, so interpret the metrics as agreement with these derived labels. Review the alignment with the data teammate before relying on the scores as evidence of physical route accuracy.

Candidate membership uses nearby sampled road geometry, then ranks shortlisted candidates by point-to-segment distance. Candidate features use distance, heading alignment, arc length, observed speed and distance rank. Macro F1 and log loss operate on binary candidate rows. Top-1 accuracy selects one arc per GPS point and counts missing references as errors. The continuity diagnostic counts adjacent predictions on the same arc or directly joined in the direction of travel. GPS gaps can span several arcs even on a valid path.

## Tuning and integration

The baseline with the best mean training-fold Macro F1 is tuned with Gaussian-process Bayesian optimisation. The first three trials are seeded samples. Later trials maximise expected improvement over a seeded proposal pool. Search ranges are in `BOUNDS` in `training/experiments.py`; logarithmic bounds are converted to actual parameter values before fitting. `results/experiments.json` records each trial, fold scores, timings, search metadata and final variant. Model selection uses training-fold scores. A separate student-test comparison shows how tuning transferred to unseen trips.

Use `training.candidates.candidates(track, nodes, arcs, k=5)` for inference. It returns arc IDs of shape `(points, candidates)` and feature rows of shape `(points, candidates, features)`. Load the trusted local `artifacts/candidate_model.joblib`, flatten feature rows, apply `predict_proba(rows)[:, 1]`, then reshape scores to match the arc ID array. Pass both arrays to the decoder. The HMM and API should call this same feature implementation or use an agreed replacement and retrain. The reference-label function is for labelled development data and scoring only.

The model comparison and search charts are generated from the recorded runs. For submission, confirm these results with your team and write your own interpretation, including the labeling limitation. Route-level sequence metrics and classifier-versus-HMM comparisons belong to the decoder integration experiment.

## Measured findings from the included run

- Logistic regression led the baselines on the predefined selection metric, training CV Macro F1 (0.837219). Extra Trees had the highest student-test Top-1 accuracy (79.96%). Histogram gradient boosting led on student-test candidate Macro F1 (0.858291) and log loss (0.209500).
- Ten tuning trials explored logistic regression regularisation strength `C` on a log scale from 0.001 to 100. The best sampled value was about 91.60. CV Macro F1 rose by 0.000025 and student-test Top-1 accuracy stayed at 79.86%. The small difference gives little evidence of a practically meaningful gain.
- The reference candidate was absent for 1.12% of student-test observations. This candidate generator limits every downstream classifier on those observations.
