## 8. Reproducibility, team contributions and AI-assistance disclosure

### 8.1 Where each result comes from

| Report section | Result | File | Produced by |
|---|---|---|---|
| 2.2 | Missing-value and integrity checks | `results/report/data_checks*.csv` | `scripts/report_data_checks.py` |
| 2.3 | EDA plots | `results/eda/*.png` | `scripts/eda.py`, `scripts/eda_plots.py` |
| 3, 4 | Five-model comparison, model tuning | `results/comparison.csv`, `results/experiments.json` | `training/experiments.py` |
| 3.4 | Final classifier metrics | `results/report/final_classifier_metrics.json` | `scripts/final_classifier_metrics.py` |
| 5 | Decoder tuning, classifier-only vs HMM, ablation | `results/context_model/*`, `results/decoder_tuning/*` | `scripts/train_context_model.py`, `scripts/tune_decoder.py` |
| 5.3 | Error analysis | `results/report/error_analysis.json` | `scripts/error_analysis.py` |
| 6 | Folium maps | `maps/*.html` | `scripts/folium_prediction_map.py` |
| All figures | Report graphs | `docs/report/figures/*.png` | `scripts/make_report_figures.py` |
| All | This document | `docs/report/` | `scripts/build_report.py` |

### 8.2 How to reproduce

```
pip install -r requirements.txt && pip install -e . && pip install -r requirements-ml.txt
python scripts/build_dataset.py "<data>/train" work/train.csv.gz
python scripts/build_dataset.py "<data>/student_test" work/student_test.csv.gz
python training/experiments.py ...                  # five-model comparison and tuning (see README-ML.md)
python scripts/train_context_model.py --train-dir "<data>/train" --train-csv work/train.csv.gz \
       --student-dir "<data>/student_test" --student-csv work/student_test.csv.gz \
       --base-model artifacts/candidate_model.joblib --trials 60 --write-artifact
python scripts/final_classifier_metrics.py ...      # same arguments as above
python scripts/make_report_figures.py && python scripts/build_report.py
python -m pytest tests -q                           # 25 tests
```
Seeds are 42 everywhere; cross-validation is GroupKFold by trip; the evaluator partition is never read by any development script.

### 8.3 Known limitations

- **Derived labels.** The reference route has no per-point labels; pointwise accuracy is measured against a monotonic alignment of the GPS points to the ordered route, which is itself a model.
- **Averaging.** Reported accuracies are means over trips; pooled over points they are slightly lower (for the final classifier alone: {{final.student_test.top1_accuracy:.3f}} pooled versus {{hmm.student_test.classifier_only.acc_covered:.3f}} mean over trips on student-test).
- **Small evaluation sets.** Student-test has 10 trips, so differences of about one point are within trip-to-trip variation; the 80-trip out-of-fold numbers are the more stable estimate.
- **Offset trips.** Training trips 00000033 and 00000054 have a constant 130-140 m GPS-to-route offset; these are kept in all reported numbers.
- **Same-tuning caveat.** The decoder was tuned on out-of-fold probabilities from training trips only; the student-test partition was not used to choose any setting.

### 8.4 Team contributions

| Member | Area | Main artefacts (see README "Credits and provenance") |
|---|---|---|
| Shwetha S | Data and features | `src/pathsnap/data`, `src/pathsnap/candidates.py`, `scripts/build_dataset.py`, `scripts/eda*.py`, data tests |
| Sruthi Vejju | Models and tuning | `training/experiments.py`, `results/comparison.csv`, `results/experiments.json`, early model artefacts, repository owner |
| Subramani B M | Decoder, API, integration, context features, decoder tuning | `src/pathsnap/hmm`, `src/pathsnap/context.py`, `src/pathsnap/pipeline.py`, `api/`, `scripts/tune_decoder.py`, `scripts/train_context_model.py`, maps, this report's build scripts |
| Shubham Singh | UI | `ui/`, `docs/ui-api-contract.md` |

<div class="team" markdown="1">
**TEAM TO WRITE (each member, 2-3 sentences):** what you personally built, one decision you made, and one thing that did not work. Everyone must be able to explain their part and the whole pipeline in the evaluation.
</div>

### 8.5 AI-assistance disclosure

<div class="team" markdown="1">
**TEAM TO CONFIRM AND COMPLETE:** the event policy requires disclosure of AI assistance. Draft for the team to edit so that it is true for *every* member: *"An AI coding assistant (Claude Code, Anthropic) was used by Subramani B M to help implement and test the HMM/Viterbi decoder, the FastAPI backend, the neighbour-point context features, the decoder-tuning and result-generation scripts, and to draft the structure of this report. All results in the report are produced by the committed scripts from the committed data files, and the team reviewed the code and results. Explanations marked 'TEAM TO WRITE' were written by the named members. [Add here which AI tools, if any, were used by Shwetha, Sruthi and Shubham, and for what.]"* Delete this paragraph's brackets and this box when final.
</div>
