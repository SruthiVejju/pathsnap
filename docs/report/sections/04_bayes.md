## 4. Bayesian hyperparameter optimisation (chosen model: HistGradientBoosting)

Code: `training/experiments.py` (function `bayesian_search`); every trial is stored in `results/experiments.json`.

1. **Hyper-parameters searched.** Learning rate and maximum number of leaf nodes.
2. **Ranges.** Learning rate in [0.025, 0.25] (sampled uniformly on that interval), max leaf nodes in [8, 64] (rounded to an integer). The number of boosting iterations was fixed at 60 and early stopping was off, so the search could not simply buy accuracy with more stages.
3. **Objective.** Maximise the mean macro-F1 over 3 folds of GroupKFold by trip on the 80 training trips (each fold's model fitted on up to 60,000 sampled rows).
4. **Number of trials.** {{tune.n}}.
5. **Search strategy.** A Gaussian-process surrogate (Matérn kernel, nu = 2.5) fitted to the trials so far; each new trial is the candidate, from a pool of 256 random points, with the highest expected improvement. The first three trials take pool points before the surrogate is fitted.
6. **Best values.** Trial {{tune.best_trial}}: learning rate {{tune.best_lr:.4f}}, max leaf nodes {{tune.best_leaves}}; mean CV macro-F1 {{tune.best_f1:.4f}} (the first trial scored {{tune.first_f1:.4f}}).

<div class="fig"><img src="figures/fig_model_tuning.png" style="width:80%"/><div class="cap">Figure 4.1. Bayesian-search trials (training partition, 3-fold GroupKFold). Mean CV macro-F1 of each trial and the best so far.</div></div>

**Did the tuned model outperform the untuned one?** Same metrics, same partitions for both:

{{@tuning_table}}

The tuned model is better on cross-validation macro-F1 by only {{gain.tune_cv:+.4f}}, which is smaller than the spread between folds ({{m.hgb_base.fold_min:.3f}} to {{m.hgb_base.fold_max:.3f}}), and it is *not* better on every student-test metric (macro-F1 and log loss are slightly worse, top-1 slightly better). The honest conclusion is that the default configuration was already close to the best this feature set allows; the improvement that mattered came from the features and the decoder (Sections 2.3 and 5). Search cost: {{tune.seconds:.1f}} s in total for the {{tune.n}} trials.

The decoder has its own Bayesian optimisation (Optuna TPE, 60 trials, seed 42), described in Section 5.

<div class="team" markdown="1">
**TEAM TO WRITE (Sruthi V):** justify the search space and the choice of a Gaussian-process / expected-improvement strategy in your own words. Why only two hyper-parameters, and what would you search with a larger budget?
</div>
