"""
src/s1_physics/compare_models.py

Builds the S1 comparison tables straight from the stored results, and asks
which failures belong to the data rather than to any one model.

Protocol section 23 item 13 requires every final table and figure to be
generated from machine-readable results rather than typed by hand, so this
emits both CSV and LaTeX ready for the Overleaf document.

Three outputs:

  table_main       one row per model: macro-F1 mean and standard deviation
                   across outer folds, accuracy, ECE, Brier, per-class F1,
                   parameter count where applicable, and wall time
  table_folds      per-outer-fold macro-F1 for every model, which is where
                   fold 0 shows up as a data property rather than a model one
  consensus        measurements that every model fails on. A recording that
                   four different algorithms cannot classify is telling you
                   something about the recording, not about the algorithms,
                   and that is the failure case section 25 asks you to present

Run from the project root:
    python src/s1_physics/compare_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from config import ARTIFACT_DIR, banner

CLASS_NAMES = {0: "drone", 1: "bird", 2: "human", 3: "reflector"}
DISPLAY = {"svm": "RBF-SVM", "rf": "Random Forest", "xgb": "XGBoost",
           "mlp": "MLP"}
ORDER = ["svm", "xgb", "rf", "mlp"]
FAIL_RECALL = 0.60          # a measurement counts as failed below this


def load_summaries() -> dict:
    out = {}
    p = ARTIFACT_DIR / "results_classical_full.json"
    if p.exists():
        d = json.loads(p.read_text())
        out.update(d["models"])
        out["_majority"] = d.get("majority_baseline")
    p = ARTIFACT_DIR / "results_mlp.json"
    if p.exists():
        out["mlp"] = json.loads(p.read_text())["model"]
    return out


def load_oof() -> dict:
    out = {}
    for p in sorted(ARTIFACT_DIR.glob("oof_predictions_*.csv")):
        key = p.stem.replace("oof_predictions_", "")
        df = pd.read_csv(p)
        df["correct"] = df["true"] == df["pred"]
        out[key] = df
    return out


def per_class_f1(df: pd.DataFrame) -> dict:
    out = {}
    for c, name in CLASS_NAMES.items():
        tp = int(((df["true"] == c) & (df["pred"] == c)).sum())
        fp = int(((df["true"] != c) & (df["pred"] == c)).sum())
        fn = int(((df["true"] == c) & (df["pred"] != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[name] = round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0
    return out


def table_main(summaries: dict, oof: dict) -> pd.DataFrame:
    rows = []
    maj = summaries.get("_majority")
    if maj:
        rows.append({"model": "Majority baseline",
                     "macro_f1": round(maj["mean_macro_f1"], 4),
                     "std": round(maj["std_macro_f1"], 4)})
    for key in ORDER:
        s = summaries.get(key)
        if s is None:
            continue
        row = {"model": DISPLAY[key],
               "macro_f1": round(s["mean_macro_f1"], 4),
               "std": round(s["std_macro_f1"], 4),
               "accuracy": round(s["accuracy"], 4),
               "ece": round(s["ece"], 4),
               "brier": round(s["brier_score"], 4),
               "params": s.get("parameter_count", ""),
               "minutes": round(s["wall_seconds"] / 60, 1)}
        if key in oof:
            row.update({f"f1_{k}": v for k, v in per_class_f1(oof[key]).items()})
            d = oof[key]
            pcols = [f"p_{CLASS_NAMES[c]}" for c in range(4)]
            order = np.argsort(-d[pcols].to_numpy(), axis=1)
            row["top2"] = round(float(np.mean(
                [t in r[:2] for t, r in zip(d["true"], order)])), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def table_folds(summaries: dict) -> pd.DataFrame:
    rows = []
    for key in ORDER:
        s = summaries.get(key)
        if s is None:
            continue
        row = {"model": DISPLAY[key]}
        for i, v in enumerate(s["outer_fold_scores"]):
            row[f"fold{i}"] = v
        row["mean"] = round(s["mean_macro_f1"], 4)
        row["std"] = round(s["std_macro_f1"], 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        cols = [c for c in df.columns if c.startswith("fold")]
        df.loc[len(df)] = {"model": "mean over models",
                           **{c: round(df[c].mean(), 4) for c in cols},
                           "mean": "", "std": ""}
    return df


def measurement_recall(oof: dict) -> pd.DataFrame:
    """Recall per measurement per model, wide format."""
    frames = []
    for key, df in oof.items():
        g = (df.groupby(["measurement_id", "outer_fold"])
               .agg(n=("correct", "size"), r=("correct", "mean"))
               .reset_index().rename(columns={"r": key}))
        frames.append(g.set_index(["measurement_id", "outer_fold", "n"])[key])
    wide = pd.concat(frames, axis=1).reset_index()
    man = pd.read_csv(ARTIFACT_DIR / "dataset_manifest.csv")
    meta = (man.groupby("measurement_id")
               .agg(label_raw=("label_raw", "first"),
                    class_name=("class_name", "first"),
                    range_m=("range_m", "median"),
                    duration_s=("time_s", "max")).reset_index())
    wide = wide.merge(meta, on="measurement_id", how="left")
    keys = [k for k in oof]
    wide["worst"] = wide[keys].min(axis=1).round(4)
    wide["best"] = wide[keys].max(axis=1).round(4)
    wide["spread"] = (wide["best"] - wide["worst"]).round(4)
    for k in keys:
        wide[k] = wide[k].round(4)
    return wide


# ---------------------------------------------------------------------------
# Paper-ready LaTeX. pandas .to_latex() is fine as a record but unusable in a
# two-column IEEE paper: thirteen columns, six decimals and no booktabs. These
# three writers emit exactly the tables the report includes with \\input, so no
# number is ever retyped (project statement, instruction 13). Best value per
# column is bolded automatically: max for macro-F1, accuracy and top-2, min for
# ECE, Brier and the cost columns.
# ---------------------------------------------------------------------------

def _fmt(v, nd=4, bold=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    txt = f"{v:.{nd}f}".lstrip("0") if isinstance(v, float) and abs(v) < 1 \
        else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    return f"\\textbf{{{txt}}}" if bold else txt


def _best(vals, lower_is_better=False):
    vals = [v for v in vals if isinstance(v, float) and not np.isnan(v)]
    if not vals:
        return None
    return min(vals) if lower_is_better else max(vals)


def write_paper_main(t: pd.DataFrame, path: Path) -> None:
    """Accuracy and calibration, five models, one column wide."""
    body = t[t["model"] != "Majority baseline"]
    best = {"macro_f1": _best(body["macro_f1"].tolist()),
            "accuracy": _best(body["accuracy"].tolist()),
            "ece": _best(body["ece"].tolist(), True),
            "brier": _best(body["brier"].tolist(), True),
            "top2": _best(body.get("top2", pd.Series(dtype=float)).tolist())}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Clean accuracy and calibration, mean $\pm$ s.d. over "
             r"five outer folds. Generated from the stored results by "
             r"\texttt{compare\_models.py}.}",
             r"\label{tab:main}", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & Macro-F1 & Acc. & ECE & Brier & Top-2 \\", r"\midrule"]
    for _, r in t.iterrows():
        cells = [r["model"]]
        f1 = r.get("macro_f1")
        sd = r.get("std")
        f1txt = _fmt(f1, 4, f1 == best["macro_f1"])
        if isinstance(sd, float) and not np.isnan(sd):
            f1txt += r" $\pm$ " + _fmt(sd, 4)
        cells.append(f1txt)
        for col, low in (("accuracy", False), ("ece", True), ("brier", True),
                         ("top2", False)):
            v = r.get(col)
            cells.append(_fmt(v, 4, v == best.get(col)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_paper_folds(t: pd.DataFrame, oof: dict, path: Path) -> None:
    """Per-fold macro-F1 beside per-class F1, one column wide."""
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Per-fold macro-F1 and pooled per-class F1. Generated "
             r"from the stored results by \texttt{compare\_models.py}.}",
             r"\label{tab:fold}", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lccccc|cccc}", r"\toprule",
             r"& \multicolumn{5}{c|}{Outer fold} & \multicolumn{4}{c}{Class F1} \\",
             r"Model & 0 & 1 & 2 & 3 & 4 & Dro. & Bird & Hum. & Refl. \\",
             r"\midrule"]
    for _, r in t.iterrows():
        if str(r["model"]).startswith("mean"):
            continue
        key = next((k for k, v in DISPLAY.items() if v == r["model"]), None)
        cells = [r["model"]] + [_fmt(r[f"fold{i}"], 3) for i in range(5)]
        pc = per_class_f1(oof[key]) if key in oof else {}
        for name in ("drone", "bird", "human", "reflector"):
            cells.append(_fmt(pc.get(name), 3))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_paper_cost(summaries: dict, path: Path) -> None:
    """Section 18 cost table, including the MLP latency from its own job."""
    lat = {}
    p = ARTIFACT_DIR / "results_mlp_latency.json"
    if p.exists():
        d = json.loads(p.read_text())
        for dev, r in d.get("devices", {}).items():
            lat[dev] = r
    rows = []
    for key in ORDER:
        s = summaries.get(key)
        if s is None:
            continue
        if key == "mlp":
            size = f"{s.get('parameter_count', '')} weights"
            for dev in ("cpu", "cuda"):
                if dev in lat:
                    rows.append((f"MLP, {dev.upper()}", size,
                                 lat[dev]["mean_latency_ms"],
                                 lat[dev]["p95_latency_ms"],
                                 lat[dev].get("worst_repeat_spread_ms"),
                                 s["wall_seconds"] / 60))
            if not lat:
                rows.append(("MLP", size, None, None, None,
                             s["wall_seconds"] / 60))
        else:
            sv = [f.get("n_support_vectors") for f in s.get("per_fold", [])]
            sv = [v for v in sv if v]
            if sv:
                size = f"{round(float(np.mean(sv))):,} SVs"
            else:
                # Depth, not tree count, drives inference cost, so name both.
                sel = (s.get("selected_params_per_fold") or [{}])[0]
                n_tree = sel.get("n_estimators", 500)
                depth = sel.get("max_depth", None)
                size = (f"{n_tree} trees, depth {depth}" if depth
                        else f"{n_tree} trees, no cap")
            rows.append((DISPLAY[key], size, s["mean_latency_ms"],
                         s["p95_latency_ms"], s.get("latency_mean_spread_ms"),
                         s["wall_seconds"] / 60))
    rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 0))
    fastest = _best([r[2] for r in rows], True)
    quickest = _best([r[5] for r in rows], True)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Computational cost, Section 18 items. Batch-one "
             r"latency on one Argon node, 100 warm-up and 1,000 timed "
             r"inferences, five repeats. Generated from the stored results by "
             r"\texttt{compare\_models.py}.}",
             r"\label{tab:cost}", r"\footnotesize",
             r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Model & Size & Lat. & p95 & Spread & Train \\",
             r" & & (ms) & (ms) & (ms) & (min) \\", r"\midrule"]
    for name, size, mean, p95, spread, mins in rows:
        lines.append(" & ".join([
            name, size,
            _fmt(mean, 3, mean == fastest),
            _fmt(p95, 3),
            _fmt(spread, 4),
            _fmt(mins, 1, mins == quickest)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    print(banner(), "\n")
    pd.set_option("display.width", 220, "display.max_columns", 60)

    summaries, oof = load_summaries(), load_oof()
    if not summaries:
        raise SystemExit("no results JSON found; run the trainers first")
    keys = [k for k in ORDER if k in oof]

    main_t = table_main(summaries, oof)
    print("=== Table 1: model comparison ===")
    print(main_t.to_string(index=False))
    main_t.to_csv(ARTIFACT_DIR / "table_main.csv", index=False)
    (ARTIFACT_DIR / "table_main.tex").write_text(
        main_t.to_latex(index=False, escape=True,
                        caption="S1 model comparison, nested 5x5 "
                                "measurement-grouped cross-validation.",
                        label="tab:s1_main"))

    folds_t = table_folds(summaries)
    print("\n=== Table 2: per outer fold ===")
    print(folds_t.to_string(index=False))
    folds_t.to_csv(ARTIFACT_DIR / "table_folds.csv", index=False)
    (ARTIFACT_DIR / "table_folds.tex").write_text(
        folds_t.to_latex(index=False, escape=True,
                         caption="Macro-F1 per outer fold.",
                         label="tab:s1_folds"))

    write_paper_main(main_t, ARTIFACT_DIR / "paper_table_main.tex")
    write_paper_folds(folds_t, oof, ARTIFACT_DIR / "paper_table_folds.tex")
    write_paper_cost(summaries, ARTIFACT_DIR / "paper_table_cost.tex")
    print("\nWrote paper_table_main.tex, paper_table_folds.tex, "
          "paper_table_cost.tex (these are what the report \\inputs)")

    if not keys:
        print("\nno out-of-fold predictions found; skipping failure analysis")
        return

    mr = measurement_recall(oof)

    print(f"\n=== Consensus failures: every model below recall {FAIL_RECALL} ===")
    cons = mr[(mr[keys] < FAIL_RECALL).all(axis=1)].sort_values("worst")
    cols = ["measurement_id", "label_raw", "outer_fold", "n"] + keys + ["spread"]
    if len(cons):
        print(cons[cols].to_string(index=False))
        print(f"\n{len(cons)} of {len(mr)} measurements "
              f"({100*len(cons)/len(mr):.1f}%) defeat all models. These carry "
              f"{int(cons['n'].sum())} segments "
              f"({100*cons['n'].sum()/mr['n'].sum():.1f}% of the dataset).")
    else:
        print("none")

    print(f"\n=== Model-specific failures: some model fails, another succeeds ===")
    split = mr[(mr[keys].min(axis=1) < FAIL_RECALL)
               & (mr[keys].max(axis=1) >= 0.85)].sort_values("spread",
                                                             ascending=False)
    print(split[cols].head(10).to_string(index=False) if len(split) else "none")

    print("\n=== Per-class mean measurement recall ===")
    print(mr.groupby("class_name")[keys].mean().round(4).to_string())

    print("\n=== Corner reflector recordings, recall against range ===")
    cr = mr[mr["class_name"] == "reflector"].sort_values("range_m")
    print(cr[["measurement_id", "range_m", "duration_s", "n"] + keys]
          .to_string(index=False))

    mr.to_csv(ARTIFACT_DIR / "measurement_recall.csv", index=False)
    print(f"\nWrote table_main.csv/.tex, table_folds.csv/.tex, "
          f"measurement_recall.csv")


if __name__ == "__main__":
    main()
