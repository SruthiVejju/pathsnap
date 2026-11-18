"""Build the technical report (HTML + PDF) from docs/report/sections/*.md and the committed result files.

Every number in the report is a {{token}} resolved from a file under results/ (so the report is verifiable against the
repository). Narrative that only the team can write is marked with a "TEAM TO WRITE" box.

  python scripts/build_report.py            # writes docs/report/report.html and PathSnap_Technical_Report.pdf
  python scripts/build_report.py --no-pdf   # HTML only

Needs: pip install markdown;  Google Chrome (headless) for the PDF step.
Token syntax:  {{a.b.c}}  {{a.b.c:.3f}}  {{@table_name}}
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys

import markdown
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT = os.path.join(ROOT, "docs", "report")
R = lambda *p: os.path.join(ROOT, "results", *p)
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

NAMES = {"logistic": "Logistic regression", "gaussian_nb": "Gaussian naive Bayes", "decision_tree": "Decision tree",
         "extra_trees": "Extra trees", "hgb_base": "HistGradientBoosting (default)", "hgb_tuned": "HistGradientBoosting (tuned)"}
FAMILY = {"logistic": "Linear / probabilistic", "gaussian_nb": "Probabilistic (generative)", "decision_tree": "Tree-based",
          "extra_trees": "Ensemble (randomised trees)", "hgb_base": "Boosting", "hgb_tuned": "Boosting"}
CONFIG = {
    "logistic": "StandardScaler + LogisticRegression(C={p0:g}, max_iter=300)",
    "gaussian_nb": "GaussianNB(var_smoothing={p0:g})",
    "decision_tree": "DecisionTree(max_depth={p0:.0f}, min_samples_leaf={p1:.0f})",
    "extra_trees": "ExtraTrees(32 trees, max_depth={p0:.0f}, min_samples_leaf={p1:.0f})",
    "hgb_base": "HistGB(learning_rate={p0:.3f}, max_leaf_nodes={p1:.0f}, max_iter=60)",
    "hgb_tuned": "HistGB(learning_rate={p0:.3f}, max_leaf_nodes={p1:.0f}, max_iter=60)",
}


def load_context():
    comp = pd.read_csv(R("comparison.csv"))
    keys = ["logistic", "gaussian_nb", "decision_tree", "extra_trees", "hgb_base", "hgb_tuned"]
    m = {}
    for k, (_, row) in zip(keys, comp.iterrows()):
        p = ast.literal_eval(row["parameters"]) if isinstance(row["parameters"], str) else row["parameters"]
        folds = ast.literal_eval(row["cv_fold_macro_f1"])
        m[k] = dict(row.to_dict(), name=NAMES[k], family=FAMILY[k], folds=folds, fold_min=min(folds), fold_max=max(folds),
                    gap=row["cv_macro_f1"] - row["macro_f1"], p0=p[0], p1=p[1] if len(p) > 1 else None,
                    config=CONFIG[k].format(p0=p[0], p1=p[1] if len(p) > 1 else 0))
    exp = json.load(open(R("experiments.json")))
    trials = exp["trials"]
    best = max(trials, key=lambda t: t["mean_cv_macro_f1"])
    fin = json.load(open(R("report", "final_classifier_metrics.json")))
    ctxm = json.load(open(R("context_model", "summary.json")))
    old = json.load(open(R("decoder_tuning", "summary.json")))
    hmm_cfg = json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json")))
    dc = pd.read_csv(R("report", "data_checks.csv"), index_col=0)
    per = pd.read_csv(R("report", "data_checks_per_trip.csv"), dtype={"trip_id": str})
    zones = sorted({int((lon + 180) // 6) + 1 for lon in per["lon_min"]} | {int((lon + 180) // 6) + 1 for lon in per["lon_max"]})
    ptc = pd.read_csv(R("context_model", "per_trip_comparison.csv"), dtype={"trip_id": str})
    tr = ptc[ptc.partition == "train_oof"]
    a = tr[tr.variant == "classifier_only"].set_index("trip_id").acc_covered
    b = tr[tr.variant == "hmm_tuned"].set_index("trip_id").acc_covered
    delta = (b - a).sort_values()
    weak = tr[tr.variant == "hmm_tuned"].set_index("trip_id").acc_covered.sort_values().head(5)
    st = ptc[ptc.partition == "student_test"]
    sa = st[st.variant == "classifier_only"].set_index("trip_id").acc_covered
    sb = st[st.variant == "hmm_tuned"].set_index("trip_id").acc_covered
    return dict(
        m=m, final=fin, hmm=ctxm["summary"], hmm_run=ctxm, old=old["summary"], hmm_cfg=hmm_cfg["params"],
        data={"train": dc.loc["train"].to_dict(), "student": dc.loc["student_test"].to_dict(), "utm_zones": len(zones)},
        sens=sensitivity(),
        err=json.load(open(R("report", "error_analysis.json"))),
        gain=dict(hmm_train=100 * (ctxm["summary"]["train_oof"]["hmm_tuned"]["acc_covered"] - ctxm["summary"]["train_oof"]["classifier_only"]["acc_covered"]),
                  hmm_student=100 * (ctxm["summary"]["student_test"]["hmm_tuned"]["acc_covered"] - ctxm["summary"]["student_test"]["classifier_only"]["acc_covered"]),
                  feat_clf_train=100 * (ctxm["summary"]["train_oof"]["classifier_only"]["acc_covered"] - old["summary"]["train_oof"]["classifier_only"]["acc_covered"]),
                  feat_final_train=100 * (ctxm["summary"]["train_oof"]["hmm_tuned"]["acc_covered"] - old["summary"]["train_oof"]["hmm_tuned"]["acc_covered"]),
                  feat_final_student=100 * (ctxm["summary"]["student_test"]["hmm_tuned"]["acc_covered"] - old["summary"]["student_test"]["hmm_tuned"]["acc_covered"]),
                  tune_cv=m["hgb_tuned"]["cv_macro_f1"] - m["hgb_base"]["cv_macro_f1"]),
        tune=dict(n=len(trials), best_trial=best["trial"], best_lr=best["parameters"][0], best_leaves=best["parameters"][1],
                  best_f1=best["mean_cv_macro_f1"], first_f1=trials[0]["mean_cv_macro_f1"], meta=exp["metadata"],
                  seconds=exp["metadata"].get("tuning_seconds", float("nan"))),
        hurt=dict(weak=", ".join(f"{t} ({100 * v:.1f}%)" for t, v in weak.items()), n=int((delta < 0).sum()), of=int(len(delta)), worst=", ".join(f"{t} ({100 * d:+.1f} points)" for t, d in delta.head(4).items()),
                  best=", ".join(f"{t} ({100 * d:+.1f} points)" for t, d in delta.tail(3).items()),
                  student_hurt=int((sb - sa < 0).sum()), student_better=int((sb - sa > 0).sum()), student_of=int(len(sb)),
                  student_best=str((sb - sa).idxmax()), student_best_gain=100 * float((sb - sa).max())),
    )


def sensitivity():
    """How flat is the decoder objective? Settings of all trials within 0.1 point of the best (training, out-of-fold)."""
    t = pd.read_csv(R("context_model", "trials.csv"))
    near = t[t["value"] >= t["value"].max() - 0.001]
    g = lambda c: (float(near[c].min()), float(near[c].max()))
    return dict(n=int(len(near)), of=int(len(t)), best=float(t["value"].max()), gap=g("params_gap_s"), hop=g("params_hop_penalty"),
                uturn=g("params_uturn_penalty"), hops=g("params_max_hops"))


def lookup(ctx, path):
    cur = ctx
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, (list, tuple)) else cur[part]
    return cur


# ----------------------------------------------------------------------------- tables
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tbl_models(ctx):
    rows = []
    for k, r in ctx["m"].items():
        rows.append([r["name"], r["family"], r["config"], f"{r['cv_macro_f1']:.3f}", f"{r['top1_accuracy']:.3f}",
                     f"{r['direct_route_continuity']:.3f}", f"{r['macro_f1']:.3f}", f"{r['log_loss']:.3f}",
                     f"{r['inference_seconds']:.3f}"])
    f = ctx["final"]
    rows.append(["**Final: HistGB, 23 features**", "Boosting", "same hyper-parameters as tuned HistGB; 11 base + 12 context features; all training rows",
                 f"{f['cv']['macro_f1']:.3f} (5-fold)", f"{f['student_test']['top1_accuracy']:.3f}", "N/A (see note)",
                 f"{f['student_test']['macro_f1']:.3f}", f"{f['student_test']['log_loss']:.3f}", f"{f['student_test']['inference_seconds']:.3f}"])
    return md_table(["Model", "Family", "Key configuration", "CV macro-F1", "Top-1 acc.", "Route cont.",
                     "Test macro-F1", "Log loss", "Infer. time (s)"], rows)


def tbl_folds(ctx):
    return md_table(["Model", "Fold 1", "Fold 2", "Fold 3", "Mean", "Student-test macro-F1", "CV minus student"],
                    [[r["name"], *[f"{x:.3f}" for x in r["folds"]], f"{r['cv_macro_f1']:.3f}", f"{r['macro_f1']:.3f}", f"{r['gap']:+.3f}"]
                     for r in ctx["m"].values()])


def tbl_tuning(ctx):
    u, t = ctx["m"]["hgb_base"], ctx["m"]["hgb_tuned"]
    rows = [["Cross-validation macro-F1", f"{u['cv_macro_f1']:.4f}", f"{t['cv_macro_f1']:.4f}", f"{t['cv_macro_f1'] - u['cv_macro_f1']:+.4f}"],
            ["Student-test top-1 accuracy", f"{u['top1_accuracy']:.4f}", f"{t['top1_accuracy']:.4f}", f"{t['top1_accuracy'] - u['top1_accuracy']:+.4f}"],
            ["Student-test macro-F1", f"{u['macro_f1']:.4f}", f"{t['macro_f1']:.4f}", f"{t['macro_f1'] - u['macro_f1']:+.4f}"],
            ["Student-test log loss", f"{u['log_loss']:.4f}", f"{t['log_loss']:.4f}", f"{t['log_loss'] - u['log_loss']:+.4f}"],
            ["Inference time (232,769 rows)", f"{u['inference_seconds']:.3f} s", f"{t['inference_seconds']:.3f} s", f"{t['inference_seconds'] - u['inference_seconds']:+.3f} s"],
            ["Training time (60,000-row fit)", f"{u['train_seconds']:.3f} s", f"{t['train_seconds']:.3f} s", f"{t['train_seconds'] - u['train_seconds']:+.3f} s"]]
    return md_table(["Metric (same partitions and metrics for both)", "Untuned (lr 0.1, 31 leaves)", "Tuned", "Change"], rows)


def pct(x):
    return f"{100 * x:.1f}%"


def tbl_hmm(ctx):
    rows = []
    for part, label in [("train_oof", "80 training trips (out-of-fold)"), ("student_test", "10 student-test trips")]:
        for key, name in [("classifier_only", "Classifier only (top-1 per point)"), ("hmm_default", "Classifier + HMM, default settings"),
                          ("hmm_tuned", "Classifier + HMM, tuned")]:
            s = ctx["hmm"][part][key]
            rows.append([label, name, pct(s["acc_covered"]), f"{s['continuity']:.3f}"])
    return md_table(["Partition", "Variant", "Pointwise accuracy (covered points)", "Route continuity"], rows)


def tbl_ablation(ctx):
    o, n = ctx["old"], ctx["hmm"]
    rows = [
        ["1", "11 base features, classifier only", pct(o["train_oof"]["classifier_only"]["acc_covered"]), pct(o["student_test"]["classifier_only"]["acc_covered"])],
        ["2", "+ HMM, default settings", pct(o["train_oof"]["hmm_default"]["acc_covered"]), pct(o["student_test"]["hmm_default"]["acc_covered"])],
        ["3", "+ Bayesian-tuned HMM (hop penalty, unreachable penalty, max hops, gap)", pct(o["train_oof"]["hmm_tuned"]["acc_covered"]), pct(o["student_test"]["hmm_tuned"]["acc_covered"])],
        ["4", "23 features (adds context features), classifier only", pct(n["train_oof"]["classifier_only"]["acc_covered"]), pct(n["student_test"]["classifier_only"]["acc_covered"])],
        ["5", "+ HMM, default settings", pct(n["train_oof"]["hmm_default"]["acc_covered"]), pct(n["student_test"]["hmm_default"]["acc_covered"])],
        ["6", "+ tuned HMM including the U-turn penalty (**final system**)", pct(n["train_oof"]["hmm_tuned"]["acc_covered"]), pct(n["student_test"]["hmm_tuned"]["acc_covered"])],
    ]
    return md_table(["Step", "System", "Training trips (out-of-fold)", "Student-test trips"], rows)


def tbl_data(ctx):
    d = ctx["data"]
    keys = [("trips", "Trips", "{:.0f}"), ("gps_points", "GPS points", "{:,.0f}"), ("median_points_per_trip", "Median GPS points per trip", "{:,.0f}"),
            ("nodes_median", "Median road nodes per trip", "{:,.0f}"), ("arcs_median", "Median road arcs per trip", "{:,.0f}"),
            ("route_arcs_median", "Median reference-route arcs per trip", "{:,.0f}"), ("nan_or_inf_values", "NaN / inf values in .track", "{:.0f}"),
            ("nan_or_inf_nodes", "NaN / inf values in .nodes", "{:.0f}"), ("nonmonotonic_timestamps", "Timestamps going backwards", "{:.0f}"),
            ("duplicate_timestamps", "Duplicate timestamps", "{:.0f}"), ("median_dt_s", "Median time step (s)", "{:.1f}"),
            ("gaps_over_30s", "Time gaps over 30 s", "{:.0f}"), ("trips_with_gap_over_30s", "Trips containing a gap over 30 s", "{:.0f}"),
            ("max_dt_s", "Largest time gap (s)", "{:,.0f}"), ("arcs_bad_node_ref", "Arcs referencing a missing node", "{:.0f}"),
            ("route_bad_arc_ref", "Route entries referencing a missing arc", "{:.0f}")]
    return md_table(["Check", "Training (80 trips)", "Student-test (10 trips)"], [[lab, fmt.format(d["train"][k]), fmt.format(d["student"][k])] for k, lab, fmt in keys])


def tbl_hmm_params(ctx):
    p = ctx["hmm_cfg"]
    meaning = {"emission_weight": "scales the classifier log-probability (fixed at 1; only ratios to the penalties matter)",
               "hop_penalty": "cost per road-network step between consecutive arcs", "unreachable_penalty": "cost when the next arc is not reachable within max_hops (soft, never forbidden)",
               "max_hops": "longest arc-to-arc path considered reachable", "gap_s": "time gap above which the chain is cut and decoded independently",
               "uturn_penalty": "cost of turning around onto the opposite-direction twin of the previous arc"}
    return md_table(["Parameter", "Tuned value", "Meaning"], [[k, f"{v:.4g}", meaning[k]] for k, v in p.items()])


def tbl_confusion(ctx):
    (tn, fp), (fn, tp) = ctx["final"]["student_test"]["confusion_matrix"]
    return md_table(["", "Predicted: not true arc", "Predicted: true arc"],
                    [["Actual: not true arc", f"{tn:,}", f"{fp:,}"], ["Actual: true arc", f"{fn:,}", f"{tp:,}"]])


TABLES = {"model_table": tbl_models, "fold_table": tbl_folds, "tuning_table": tbl_tuning, "hmm_table": tbl_hmm,
          "ablation_table": tbl_ablation, "data_table": tbl_data, "hmm_params_table": tbl_hmm_params, "confusion_table": tbl_confusion}


def render(text, ctx):
    def sub(match):
        tok, spec = match.group(1), match.group(2)
        if tok.startswith("@"):
            return TABLES[tok[1:]](ctx)
        v = lookup(ctx, tok)
        if spec == "pct":
            return f"{100 * v:.1f}"
        return format(v, spec) if spec else str(v)
    return re.sub(r"\{\{([@\w\.]+)(?::([^}]+))?\}\}", sub, text)


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; @bottom-center { content: counter(page); font: 9pt sans-serif; color: #52514e; } }
body { font: 10.5pt/1.5 'Helvetica Neue', Arial, sans-serif; color: #0b0b0b; max-width: 178mm; margin: 0 auto; }
h1 { font-size: 22pt; margin: 0 0 4pt; } h2 { font-size: 15pt; margin: 22pt 0 6pt; border-bottom: 1px solid #d5d4cf; padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; break-after: avoid; } h4 { font-size: 10.5pt; margin: 10pt 0 2pt; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8pt; break-inside: avoid; table-layout: auto; }
th, td { border: 1px solid #cfcdc7; padding: 3pt 4pt; text-align: left; vertical-align: top; overflow-wrap: break-word; } th { background: #f1f0ec; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; break-inside: avoid; } figure, .fig { break-inside: avoid; margin: 8pt 0; }
.cap { font-size: 8.8pt; color: #52514e; text-align: center; margin: 0 0 10pt; }
code { font: 9pt Menlo, monospace; background: #f1f0ec; padding: 0 2pt; } pre { background: #f1f0ec; padding: 6pt; font-size: 8.4pt; overflow: hidden; break-inside: avoid; }
.team { background: #fff4cc; border: 1.5px solid #e0a800; border-radius: 4px; padding: 6pt 9pt; margin: 8pt 0; break-inside: avoid; }
.shot { background: #eaf2fd; border: 1.5px dashed #2a78d6; border-radius: 4px; padding: 14pt 9pt; margin: 8pt 0; text-align: center; break-inside: avoid; }
.note { background: #f1f0ec; border-left: 3px solid #52514e; padding: 4pt 8pt; margin: 8pt 0; font-size: 9.4pt; break-inside: avoid; }
.draft { color: #b3261e; font-weight: 700; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    ctx = load_context()
    secs = sorted(f for f in os.listdir(os.path.join(REPORT, "sections")) if f.endswith(".md"))
    body = "\n\n".join(render(open(os.path.join(REPORT, "sections", f)).read(), ctx) for f in secs)
    html = markdown.markdown(body, extensions=["tables", "fenced_code", "md_in_html", "sane_lists"])
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap technical report</title><style>{CSS}</style></head><body>{html}</body></html>"
    out_html = os.path.join(REPORT, "report.html")
    open(out_html, "w").write(page)
    print("wrote", out_html)
    if args.no_pdf:
        return
    pdf = os.path.join(REPORT, "PathSnap_Technical_Report.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "--virtual-time-budget=15000", f"file://{out_html}"], check=True, capture_output=True)
    print("wrote", pdf)


if __name__ == "__main__":
    sys.exit(main())
