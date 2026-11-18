# PathSnap model experiments

Run from the directory containing `training/` with Python 3.12.

```sh
python -m pip install -r requirements-ml.txt
python -m training.experiments "/path/to/Student folder.zip"
python -m training.plots
```

The default run compares logistic regression, Gaussian naive Bayes, a decision tree, Extra Trees and histogram gradient boosting. All five use the same features and three trip-grouped folds from the 80 training trips. Each fit uses a deterministic sample of up to 60,000 candidate rows drawn from its training trips. Validation and student-test scoring use every available observation. The 10 student-test trips are scored separately. The evaluator partition is outside this experiment.

## Labels and evaluation

The reference file lists a sequence of arcs for the entire trip. The experiment aligns GPS observations monotonically to that ordered route using a dynamic-programming distance objective, then labels a candidate positive when its arc matches the aligned arc. This remains an approximate label and can be wrong near parallel roads, crossings, GPS dropouts, or bad reference data. Interpret metrics as agreement with the supplied route, not verified physical route accuracy.

Candidate membership uses exact point-to-segment distance in metres after per-trip UTM projection, retaining arcs within 60 m and up to 25 candidates per observation. The shared `build_candidates` function does not accept or read a route; labels are added separately for training only. The 11 candidate feature columns are listed in `FEATURE_COLUMNS` in `src/pathsnap/candidates.py`. Macro F1 and log loss operate on binary candidate rows. Top-1 accuracy selects one arc per GPS point and counts missing references as errors. The continuity diagnostic counts adjacent predictions on the same arc or directly joined in the direction of travel. GPS gaps can span several arcs even on a valid path.

## Tuning and integration

The baseline with the best mean training-fold Macro F1 is tuned with Gaussian-process Bayesian optimisation. The first three trials are seeded samples. Later trials maximise expected improvement over a seeded proposal pool. Search ranges are in `BOUNDS` in `training/experiments.py`; logarithmic bounds are converted to actual parameter values before fitting. `results/experiments.json` records each trial, fold scores, timings, search metadata and final variant. Model selection uses training-fold scores. A separate student-test comparison shows how tuning transferred to unseen trips.

Candidate generation is implemented in `src/pathsnap/candidates.py`. Call `build_candidates(trip)` without a route to get the point/arc candidate table; it uses a 60 m radius and keeps up to 25 candidates per observation. The ordered 11 model feature names are defined once in `FEATURE_COLUMNS`. Use `add_labels(frame, trip)` only for labelled training and development data; its monotonic route alignment is not part of inference. The model trainer imports these functions and stores `feature_columns`, radius, candidate cap, and alignment method in `artifacts/model_config.json`.

The model comparison and search charts are generated from the recorded runs. For submission, confirm these results with your team and write your own interpretation, including the labeling limitation. Route-level sequence metrics and classifier-versus-HMM comparisons belong to the decoder integration experiment.

## Previous experiment results

The committed comparison and model artifact were generated with the former 5-feature,
top-5 candidate pipeline. They are retained as history but are not compatible with the
current 11-feature Part A pipeline. Rerun the experiments and regenerate the artifact
before using it for inference.

- Logistic regression led the baselines on the predefined selection metric, training CV Macro F1 (0.837219). Extra Trees had the highest student-test Top-1 accuracy (79.96%). Histogram gradient boosting led on student-test candidate Macro F1 (0.858291) and log loss (0.209500).
- Ten tuning trials explored logistic regression regularisation strength `C` on a log scale from 0.001 to 100. The best sampled value was about 91.60. CV Macro F1 rose by 0.000025 and student-test Top-1 accuracy stayed at 79.86%. The small difference gives little evidence of a practically meaningful gain.
- The reference candidate was absent for 1.12% of student-test observations. This candidate generator limits every downstream classifier on those observations.
