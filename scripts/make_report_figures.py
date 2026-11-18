"""Figures for the technical report, generated ONLY from result files committed in this repository.

Inputs  results/comparison.csv, results/experiments.json (model comparison + model tuning),
        results/report/final_classifier_metrics.json (scripts/final_classifier_metrics.py),
        results/context_model/{summary.json, trials.csv, per_trip_comparison.csv} (scripts/train_context_model.py),
        results/decoder_tuning/summary.json (scripts/tune_decoder.py, 11-feature model).
Output  docs/report/figures/*.png

Colours: slots 1-3 of the validated categorical palette (blue, orange, aqua; all-pairs CVD-safe). Aqua is below 3:1
contrast on white, so every bar/point is also labelled with its value.

  python scripts/make_report_figures.py
"""
import ast
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = lambda *p: os.path.join(ROOT, "results", *p)
OUT = os.path.join(ROOT, "docs", "report", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})

LABELS = {"logistic": "Logistic regression", "gaussian_nb": "Gaussian naive Bayes", "decision_tree": "Decision tree",
          "extra_trees": "Extra trees", "hist_gradient_boosting": "HistGradientBoosting"}


def model_rows():
    df = pd.read_csv(R("comparison.csv"))
    df["label"] = [LABELS[m] + (" (tuned)" if v == "tuned" else "") for m, v in zip(df.model, df.variant)]
    df["folds"] = df["cv_fold_macro_f1"].apply(ast.literal_eval)
    return df


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def hbar(ax, labels, values, colors, fmt="{:.3f}", err=None, xlim=None, extra=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.58, xerr=err, ecolor=MUTED, capsize=3)
    ax.set_yticks(y, labels)
    extra = extra if extra is not None else [0] * len(values)
    for yy, v, ex in zip(y, values, extra):
        ax.text(v + ex + ((xlim[1] - xlim[0]) * 0.012 if xlim else 0), yy, fmt.format(v), va="center", fontsize=9, color=INK)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)


def fig_cv_f1(df):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = [ORANGE if v == "tuned" else BLUE for v in df.variant]
    lo = [m - min(f) for m, f in zip(df.cv_macro_f1, df.folds)]
    hi = [max(f) - m for m, f in zip(df.cv_macro_f1, df.folds)]
    hbar(ax, df.label, df.cv_macro_f1, colors, err=[lo, hi], xlim=(0.6, 1.0), extra=hi)
    ax.set_xlabel("Cross-validation macro-F1 (3-fold GroupKFold by trip; bars show min-max across folds)")
    save(fig, "fig_cv_macro_f1.png")


def fig_student(df):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(df))[::-1]
    h = 0.36
    ax.barh(y + h / 2, df.top1_accuracy, h, color=BLUE, label="Top-1 accuracy")
    ax.barh(y - h / 2, df.macro_f1, h, color=ORANGE, label="Macro-F1")
    for yy, a, b in zip(y, df.top1_accuracy, df.macro_f1):
        ax.text(a + 0.006, yy + h / 2, f"{a:.3f}", va="center", fontsize=8.5)
        ax.text(b + 0.006, yy - h / 2, f"{b:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y, df.label)
    ax.set_xlim(0.6, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Student-test partition (10 trips)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig_student_top1_f1.png")


def fig_logloss(df):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = [ORANGE if v == "tuned" else BLUE for v in df.variant]
    hbar(ax, df.label, df.log_loss, colors, xlim=(0, 0.55))
    ax.set_xlabel("Student-test log loss (lower is better)")
    save(fig, "fig_log_loss.png")


def fig_confusion():
    m = json.load(open(R("report", "final_classifier_metrics.json")))["student_test"]["confusion_matrix"]
    m = np.array(m)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=11,
                    color="white" if m[i, j] > m.max() * 0.5 else INK)
    ax.set_xticks([0, 1], ["not true arc", "true arc"])
    ax.set_yticks([0, 1], ["not true arc", "true arc"])
    ax.set_xlabel("Predicted (threshold 0.5)")
    ax.set_ylabel("Actual (derived label)")
    ax.spines[:].set_visible(False)
    save(fig, "fig_confusion_final.png")


def fig_hmm_stages():
    s = json.load(open(R("context_model", "summary.json")))["summary"]
    names = [("classifier_only", "Classifier only", BLUE), ("hmm_default", "Classifier + HMM (default settings)", ORANGE),
             ("hmm_tuned", "Classifier + HMM (tuned)", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True)
    for ax, part, title in zip(axes, ["train_oof", "student_test"],
                               ["80 training trips (out-of-fold)", "10 student-test trips"]):
        x = np.arange(2)
        for k, (key, lab, c) in enumerate(names):
            vals = [100 * s[part][key]["acc_covered"], 100 * s[part][key]["continuity"]]
            ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=c, label=lab)
            for xi, v in zip(x + (k - 1) * 0.26, vals):
                ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(x, ["Pointwise accuracy", "Route continuity"])
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_ylim(60, 102)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean over trips (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig_hmm_stages.png")


def fig_per_trip():
    d = pd.read_csv(R("context_model", "per_trip_comparison.csv"), dtype={"trip_id": str})
    d = d[d.partition == "student_test"]
    a = d[d.variant == "classifier_only"].set_index("trip_id")["acc_covered"]
    b = d[d.variant == "hmm_tuned"].set_index("trip_id")["acc_covered"]
    ids = list(b.sort_values().index)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(ids))
    for yy, t in zip(y, ids):
        ax.plot([a[t], b[t]], [yy, yy], color=GRID, lw=2, zorder=1)
    ax.scatter([a[t] for t in ids], y, s=46, color=BLUE, zorder=3, label="Classifier only")
    ax.scatter([b[t] for t in ids], y, s=46, color=ORANGE, zorder=3, label="Classifier + tuned HMM")
    for yy, t in zip(y, ids):
        ax.text(max(a[t], b[t]) + 0.008, yy, f"{b[t]:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y, [f"Trip {t}" for t in ids])
    ax.set_xlim(0.7, 1.04)
    ax.set_xlabel("Pointwise accuracy, student-test trips (value shown = with HMM)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig_per_trip_student.png")


def fig_decoder_trials():
    t = pd.read_csv(R("context_model", "trials.csv"))
    v = t["value"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.scatter(t["number"], v, s=22, color=BLUE, label="Trial")
    ax.plot(t["number"], np.maximum.accumulate(v), color=ORANGE, lw=2, label="Best so far")
    ax.set_xlabel("Optuna trial (TPE, seed 42)")
    ax.set_ylabel("Mean pointwise accuracy,\ntraining trips (out-of-fold)")
    ax.set_ylim(np.nanmin(v) - 0.01, v.max() + 0.006)
    ax.legend(loc="lower right")
    save(fig, "fig_decoder_trials.png")


def fig_model_tuning():
    tr = json.load(open(R("experiments.json")))["trials"]
    v = [t["mean_cv_macro_f1"] for t in tr]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(range(1, len(v) + 1), v, "o", color=BLUE, label="Trial")
    ax.plot(range(1, len(v) + 1), np.maximum.accumulate(v), color=ORANGE, lw=2, label="Best so far")
    ax.set_xlabel("Bayesian-search trial (Gaussian process, expected improvement)")
    ax.set_ylabel("Mean CV macro-F1")
    ax.legend(loc="lower right")
    save(fig, "fig_model_tuning.png")


def main():
    df = model_rows()
    fig_cv_f1(df)
    fig_student(df)
    fig_logloss(df)
    fig_confusion()
    fig_hmm_stages()
    fig_per_trip()
    fig_decoder_trials()
    fig_model_tuning()


if __name__ == "__main__":
    main()
