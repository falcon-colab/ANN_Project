"""
src/s1_physics/analyze_results.py

Everything the summary table cannot tell you, computed from the saved
out-of-fold predictions.

Covers four things the protocol requires and one hypothesis worth testing:

  section 16   top-2 prediction, confidence-threshold analysis, confident
               incorrect predictions
  section 19   class-wise error analysis
  section 25   the failure case you must present
  diagnosis    why outer fold 0 is the worst fold for all three models
  hypothesis   the drone class looks internally split between hovering and
               translating flight, from its very wide interquartile range on
               zero_doppler_ratio. If true, the misclassified drones should be
               the translating ones. This tests it directly.

Run from the project root:
    python src/s1_physics/analyze_results.py
    python src/s1_physics/analyze_results.py --model xgb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from config import ARTIFACT_DIR, banner

CLASS_NAMES = {0: "drone", 1: "bird", 2: "human", 3: "reflector"}
PROB_COLS = [f"p_{CLASS_NAMES[c]}" for c in range(4)]
CONFIDENT = 0.90


def load(model_key: str) -> pd.DataFrame:
    oof = pd.read_csv(ARTIFACT_DIR / f"oof_predictions_{model_key}.csv")
    feats = pd.read_csv(ARTIFACT_DIR / "s1_features.csv")
    cols = ["segment_uid", "zero_doppler_ratio", "doppler_spread",
            "temporal_entropy"]
    df = oof.merge(feats[cols], on="segment_uid", how="left")
    man = pd.read_csv(ARTIFACT_DIR / "dataset_manifest.csv")
    df = df.merge(man[["segment_uid", "label_raw"]], on="segment_uid", how="left")
    df["correct"] = df["true"] == df["pred"]
    df["confidence"] = df[PROB_COLS].to_numpy().max(axis=1)
    order = np.argsort(-df[PROB_COLS].to_numpy(), axis=1)
    df["top2_hit"] = [t in row[:2] for t, row in zip(df["true"], order)]
    return df


def per_class(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, name in CLASS_NAMES.items():
        tp = int(((df["true"] == c) & (df["pred"] == c)).sum())
        fp = int(((df["true"] != c) & (df["pred"] == c)).sum())
        fn = int(((df["true"] == c) & (df["pred"] != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": name, "support": int((df["true"] == c).sum()),
                     "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4)})
    return pd.DataFrame(rows)


def confusion(df: pd.DataFrame, normalise: bool = True) -> pd.DataFrame:
    m = pd.crosstab(df["true"].map(CLASS_NAMES), df["pred"].map(CLASS_NAMES),
                    dropna=False)
    m = m.reindex(index=list(CLASS_NAMES.values()),
                  columns=list(CLASS_NAMES.values()), fill_value=0)
    if normalise:
        m = (m.div(m.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(1)
    return m


def fold_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby("outer_fold"):
        pc = per_class(g).set_index("class")
        row = {"fold": k, "n": len(g),
               "macro_f1": round(pc["f1"].mean(), 4),
               "accuracy": round(g["correct"].mean(), 4)}
        for name in CLASS_NAMES.values():
            row[f"f1_{name}"] = pc.loc[name, "f1"]
            row[f"n_{name}"] = int(pc.loc[name, "support"])
        rows.append(row)
    return pd.DataFrame(rows)


def worst_measurements(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    g = df.groupby(["measurement_id", "label_raw", "outer_fold"]).agg(
        n_segments=("correct", "size"), recall=("correct", "mean")).reset_index()
    return g.sort_values("recall").head(n).assign(
        recall=lambda d: d["recall"].round(4))


def drone_split(df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """
    The hovering-versus-translating test.

    A hovering drone's body return sits at zero Doppler, so its
    zero_doppler_ratio is high. A translating drone's return moves off DC, so
    the ratio falls. If the drone class really is internally split and the
    translating ones are the hard cases, recall should rise monotonically with
    zero_doppler_ratio.
    """
    d = df[df["true"] == 0].copy()
    if d.empty:
        return pd.DataFrame()
    d["bin"] = pd.qcut(d["zero_doppler_ratio"], n_bins, duplicates="drop")
    out = d.groupby("bin", observed=True).agg(
        n=("correct", "size"),
        zd_median=("zero_doppler_ratio", "median"),
        recall=("correct", "mean"),
        confidence=("confidence", "mean")).reset_index()
    out["mistaken_for"] = [
        d.loc[(d["bin"] == b) & (~d["correct"]), "pred"].map(CLASS_NAMES)
         .value_counts().head(1).to_dict() or {"none": 0}
        for b in out["bin"]]
    for c in ("zd_median", "recall", "confidence"):
        out[c] = out[c].round(4)
    return out


def confidence_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        kept = df[df["confidence"] >= thr]
        rows.append({"threshold": thr,
                     "coverage": round(len(kept) / len(df), 4),
                     "accuracy": round(kept["correct"].mean(), 4)
                     if len(kept) else float("nan")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="svm")
    ap.add_argument("--all", action="store_true",
                    help="loop over every model with saved predictions")
    args = ap.parse_args()

    print(banner(), "\n")
    pd.set_option("display.width", 200, "display.max_columns", 40)

    keys = args.model.split(",")
    if args.all:
        keys = [p.stem.replace("oof_predictions_", "")
                for p in sorted(ARTIFACT_DIR.glob("oof_predictions_*.csv"))]

    for key in keys:
        df = load(key)
        print("=" * 78)
        print(f"{key.upper()}  |  {len(df)} out-of-fold segments")
        print("=" * 78)

        print("\n-- per class (pooled over all five outer folds) --")
        print(per_class(df).to_string(index=False))
        print(f"\nmacro-F1 {per_class(df)['f1'].mean():.4f} | "
              f"accuracy {df['correct'].mean():.4f} | "
              f"top-2 accuracy {df['top2_hit'].mean():.4f}")

        print("\n-- confusion matrix, row-normalised (% of each true class) --")
        print(confusion(df).to_string())

        print("\n-- per fold --")
        print(fold_breakdown(df).to_string(index=False))

        print("\n-- ten worst measurements by recall --")
        print(worst_measurements(df).to_string(index=False))

        print("\n-- drone recall by zero_doppler_ratio quartile --")
        ds = drone_split(df)
        if not ds.empty:
            print(ds.to_string(index=False))
            lo, hi = ds["recall"].iloc[0], ds["recall"].iloc[-1]
            trend = ("supports" if hi > lo else "does not support")
            print(f"   lowest quartile recall {lo:.4f} vs highest {hi:.4f}: "
                  f"{trend} the hovering/translating split hypothesis")

        print("\n-- confidence threshold (section 16) --")
        print(confidence_analysis(df).to_string(index=False))

        conf_wrong = df[(~df["correct"]) & (df["confidence"] >= CONFIDENT)]
        print(f"\n-- confident incorrect: {len(conf_wrong)} segments at "
              f"confidence >= {CONFIDENT} "
              f"({len(conf_wrong)/len(df)*100:.2f}% of all) --")
        if len(conf_wrong):
            print(conf_wrong.groupby([conf_wrong["true"].map(CLASS_NAMES),
                                      conf_wrong["pred"].map(CLASS_NAMES)])
                  .size().rename("n").reset_index().to_string(index=False))
        print()


if __name__ == "__main__":
    main()
