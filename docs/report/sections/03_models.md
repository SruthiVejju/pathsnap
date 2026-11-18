## 3. Model experimentation and comparison

Five model families were trained and compared with the same data, features and protocol: **3-fold GroupKFold by trip** on the 80 training trips (so no trip is in both the training and validation part of a fold), a fit on up to 60,000 sampled candidate rows, and evaluation on the 10 student-test trips. Seed 42 throughout. Code: `training/experiments.py`; raw results: `results/experiments.json`, `results/comparison.csv`. These comparisons use the 11 base features; the final model (23 features) is described at the end of this section.

### 3.1 Overall model comparison

{{@model_table}}

*Notes.* Metrics are on the student-test partition except CV macro-F1. **Top-1 accuracy** is observation-level: for each GPS point the highest-scoring candidate is compared with the derived reference arc. **Route continuity** in this table is the share of consecutive top-1 arcs that are connected in the road network (classifier only, no HMM); the final-system row shows N/A because it is defined for the full pipeline in Section 5 (the HMM-aware continuity values are in Section 5). **Inference time** is `predict_proba` over all {{final.student_test.candidate_rows:,}} student-test candidate rows. Candidate miss rate is {{m.logistic.candidate_miss_rate:.2%}} for every model (it depends on candidate generation, not on the model). Gaussian naive Bayes' very high recall ({{m.gaussian_nb.macro_recall:.3f}}) with low precision ({{m.gaussian_nb.macro_precision:.3f}}) means it over-predicts "true arc".

<div class="fig"><img src="figures/fig_cv_macro_f1.png" style="width:82%"/><div class="cap">Figure 3.1. Cross-validation macro-F1 per model (training partition, 3-fold GroupKFold by trip). Whiskers show the lowest and highest fold. Orange = the tuned model that was carried forward.</div></div>

<div class="fig"><img src="figures/fig_student_top1_f1.png" style="width:82%"/><div class="cap">Figure 3.2. Observation-level top-1 accuracy and macro-F1 on the student-test partition.</div></div>

<div class="fig"><img src="figures/fig_log_loss.png" style="width:82%"/><div class="cap">Figure 3.3. Student-test log loss (lower is better).</div></div>

Per-fold scores and the gap between cross-validation and student-test performance (used below as overfitting evidence):

{{@fold_table}}

### 3.2 Per-model analysis

None of these models produces an epoch-wise training history (they are fitted in one pass or by a fixed number of boosting stages), so no training/validation loss curves are shown; creating them would be artificial.

#### Model 1. Logistic regression (`{{m.logistic.config}}`)
**A. Understanding.** <div class="team" markdown="1">**TEAM TO WRITE (Sruthi V):** why was a linear model considered suitable here, and what are its strengths and limitations for this task? Connect your answer to the numbers in B and C.</div>
**B. Training and configuration.** Initial hyper-parameters: regularisation strength C = {{m.logistic.p0:g}}, features standardised, at most 300 iterations. Objective: the regularised negative log-likelihood (log loss with an L2 penalty); no search was run for this model. Training time on the 60,000-row fit: {{m.logistic.train_seconds:.3f}} s.
**C. Results.** Student-test top-1 accuracy {{m.logistic.top1_accuracy:.3f}}, macro-F1 {{m.logistic.macro_f1:.3f}}, log loss {{m.logistic.log_loss:.3f}}. Cross-validation macro-F1 {{m.logistic.cv_macro_f1:.3f}} (folds {{m.logistic.fold_min:.3f}} to {{m.logistic.fold_max:.3f}}); student-test macro-F1 is {{m.logistic.gap:+.3f}} below it, i.e. the model generalises about as well as it cross-validates, with no sign of overfitting but a lower ceiling than the boosted model.

#### Model 2. Gaussian naive Bayes (`{{m.gaussian_nb.config}}`)
**A. Understanding.** <div class="team" markdown="1">**TEAM TO WRITE (Sruthi V):** why a generative probabilistic model, and what does its independence assumption mean for features such as `dist_m` and `dist_rank` that are strongly related?</div>
**B. Training and configuration.** Initial hyper-parameter: variance smoothing {{m.gaussian_nb.p0:g}}. Objective: maximum-likelihood estimates of a per-class Gaussian for each feature (no iterative optimisation). Training time {{m.gaussian_nb.train_seconds:.3f}} s.
**C. Results.** Top-1 accuracy {{m.gaussian_nb.top1_accuracy:.3f}}, macro-F1 {{m.gaussian_nb.macro_f1:.3f}}, log loss {{m.gaussian_nb.log_loss:.3f}}, the weakest on every metric. Its log loss is far above the others, so its probabilities are poorly calibrated, which matters because the HMM uses them directly as emission scores. CV macro-F1 {{m.gaussian_nb.cv_macro_f1:.3f}} versus {{m.gaussian_nb.macro_f1:.3f}} on student-test ({{m.gaussian_nb.gap:+.3f}}); its fold scores vary the most ({{m.gaussian_nb.fold_min:.3f}} to {{m.gaussian_nb.fold_max:.3f}}). The shortfall looks like underfitting (a model that is too simple for correlated features) rather than overfitting.

#### Model 3. Decision tree (`{{m.decision_tree.config}}`)
**A. Understanding.** <div class="team" markdown="1">**TEAM TO WRITE (Sruthi V):** why trees, and what can a single tree capture that the logistic model cannot (think of interactions between distance, heading and rank)?</div>
**B. Training and configuration.** Initial hyper-parameters: max depth {{m.decision_tree.p0:.0f}}, minimum samples per leaf {{m.decision_tree.p1:.0f}}. Objective: greedy splits that minimise Gini impurity (scikit-learn default). Training time {{m.decision_tree.train_seconds:.3f}} s.
**C. Results.** Top-1 {{m.decision_tree.top1_accuracy:.3f}}, macro-F1 {{m.decision_tree.macro_f1:.3f}}, log loss {{m.decision_tree.log_loss:.3f}}. Its macro-F1 is higher than logistic regression's but its log loss is more than twice as high ({{m.decision_tree.log_loss:.3f}} versus {{m.logistic.log_loss:.3f}}): the tree ranks well at a 0.5 threshold, but its leaf-frequency probabilities are over-confident. CV {{m.decision_tree.cv_macro_f1:.3f}} versus student-test {{m.decision_tree.macro_f1:.3f}} ({{m.decision_tree.gap:+.3f}}).

#### Model 4. Extra trees (`{{m.extra_trees.config}}`)
**A. Understanding.** <div class="team" markdown="1">**TEAM TO WRITE (Sruthi V):** why an ensemble of randomised trees, and how does averaging many trees change the probabilities compared with one tree?</div>
**B. Training and configuration.** Initial hyper-parameters: 32 trees, max depth {{m.extra_trees.p0:.0f}}, minimum samples per leaf {{m.extra_trees.p1:.0f}}. Objective: Gini impurity with randomly drawn split thresholds; the forest probability is the mean of the trees' leaf frequencies. Training time {{m.extra_trees.train_seconds:.3f}} s; inference {{m.extra_trees.inference_seconds:.3f}} s, about the same as boosting and about 40 times slower than logistic regression ({{m.logistic.inference_seconds:.3f}} s).
**C. Results.** Top-1 {{m.extra_trees.top1_accuracy:.3f}}, macro-F1 {{m.extra_trees.macro_f1:.3f}}, log loss {{m.extra_trees.log_loss:.3f}}, which is much better calibrated than the single tree ({{m.decision_tree.log_loss:.3f}}) and the second-best model overall. CV {{m.extra_trees.cv_macro_f1:.3f}} versus {{m.extra_trees.macro_f1:.3f}} ({{m.extra_trees.gap:+.3f}}).

#### Model 5. HistGradientBoosting (`{{m.hgb_base.config}}`)
**A. Understanding.** <div class="team" markdown="1">**TEAM TO WRITE (Sruthi V):** why boosting, and why do tabular geometric features like ours suit gradient-boosted trees? Explain what "histogram" binning buys in speed and how it handles missing values (this matters for the context features, which contain NaN).</div>
**B. Training and configuration.** Initial hyper-parameters: learning rate {{m.hgb_base.p0:g}}, max leaf nodes {{m.hgb_base.p1:.0f}}, 60 boosting iterations, no early stopping. Objective: binary cross-entropy (log loss), optimised stage by stage on the gradient of the loss. Training time {{m.hgb_base.train_seconds:.3f}} s.
**C. Results.** Top-1 {{m.hgb_base.top1_accuracy:.3f}}, macro-F1 {{m.hgb_base.macro_f1:.3f}}, log loss {{m.hgb_base.log_loss:.3f}}: the best CV macro-F1 ({{m.hgb_base.cv_macro_f1:.3f}}), the best top-1 and the lowest log loss. CV versus student-test macro-F1 differ by {{m.hgb_base.gap:+.3f}}, with fold scores {{m.hgb_base.fold_min:.3f}} to {{m.hgb_base.fold_max:.3f}}. The tuned variant is analysed in Section 4.

### 3.3 Model visualisations

Figures 3.1-3.3 compare all models on cross-validation macro-F1, student-test top-1 accuracy and macro-F1, and log loss. Confusion matrix of the **final classifier** on the student-test partition (threshold 0.5; the classes are heavily imbalanced because most candidates are not the true arc):

<div class="fig"><img src="figures/fig_confusion_final.png" style="width:48%"/><div class="cap">Figure 3.4. Confusion matrix of the final classifier, student-test partition. Rows are the derived label, columns the prediction.</div></div>

{{@confusion_table}}

### 3.4 Model selection

HistGradientBoosting had the highest cross-validation macro-F1, the highest top-1 accuracy and the lowest log loss; extra trees was close behind. Log loss is weighted heavily in the choice because the decoder consumes the probabilities directly: a model with the same macro-F1 but over-confident probabilities (the decision tree) gives the HMM poorer emission scores.

**Final classifier.** After the error analysis in Section 5, the classifier was refitted with the same hyper-parameters on **23 features** (the 11 base features plus 12 neighbour-point context features) using **all** training rows instead of a 60,000-row sample. Its measured performance (script: `scripts/final_classifier_metrics.py`, output `results/report/final_classifier_metrics.json`):

| Metric | 5-fold GroupKFold (training trips) | Student-test |
|---|---|---|
| Macro-F1 | {{final.cv.macro_f1:.3f}} (folds {{final.cv.fold_macro_f1.0:.3f}}, {{final.cv.fold_macro_f1.1:.3f}}, {{final.cv.fold_macro_f1.2:.3f}}, {{final.cv.fold_macro_f1.3:.3f}}, {{final.cv.fold_macro_f1.4:.3f}}) | {{final.student_test.macro_f1:.3f}} |
| Observation-level top-1 accuracy | {{final.cv.top1_accuracy:.3f}} | {{final.student_test.top1_accuracy:.3f}} |
| Log loss | {{final.cv.log_loss:.3f}} | {{final.student_test.log_loss:.3f}} |
| Macro precision / recall | n/a | {{final.student_test.macro_precision:.3f}} / {{final.student_test.macro_recall:.3f}} |

The in-sample training macro-F1 of the shipped model is {{final.train_in_sample.macro_f1:.3f}}, compared with {{final.cv.macro_f1:.3f}} out-of-fold and {{final.student_test.macro_f1:.3f}} on student-test, so the gap between fitting and validation is small (the model has only 60 boosting stages) and no strong overfitting is visible.

<div class="team" markdown="1">
**TEAM TO WRITE (Sruthi V):** in your own words, explain the primary metric you used to select the model and why it suits PathSnap (candidate-level macro-F1 versus observation-level top-1 versus the final route accuracy), and what you would try with more time.
</div>
