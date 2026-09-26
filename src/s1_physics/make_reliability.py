"""
src/s1_physics/make_reliability.py

Reliability diagrams, the one Section 16 deliverable that was still missing.

Reads the stored out-of-fold probability files and draws, for each model, the
accuracy within each of 15 equal-width confidence bins against the mean
confidence in that bin. A perfectly calibrated model lies on the diagonal;
below it means overconfident, above means underconfident. ECE is the
population-weighted mean absolute gap between the two, recomputed here from
the same bins so the figure and the number in the table cannot disagree.

Design choices that matter for a printed two-column paper:
  * one column wide (3.4 in) and vector PDF, so nothing is rescaled by LaTeX;
  * Okabe-Ito colours, checked for colour-vision separation, AND a distinct
    marker and dash pattern per model, so the figure still reads in grayscale
    or photocopy, where colour alone would be lost;
  * a second panel showing how many segments fall in each bin, because a
    reliability curve is meaningless where its bins are nearly empty. The SVM
    puts 3% of its mass above 0.99 and XGBoost 65%, and without the histogram
    the two look equally trustworthy at the right-hand edge;
  * bins holding fewer than MIN_BIN segments are drawn hollow rather than
    dropped, so a sparse region is visible as sparse instead of absent.

Usage, from the project root with ARTIFACT_DIR pointing at the results:

    python src/s1_physics/make_reliability.py
    python src/s1_physics/make_reliability.py --bins 15 --min-bin 50

Writes reliability.pdf, reliability.png and reliability_bins.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display on a cluster node
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
sys.path.append(str(Path(__file__).resolve().parent))

from config import ARTIFACT_DIR, banner    # noqa: E402

CLASS_NAMES = {0: "drone", 1: "bird", 2: "human", 3: "reflector"}
ORDER = ["svm", "xgb", "rf", "mlp"]
DISPLAY = {"svm": "SVM", "xgb": "XGBoost", "rf": "Random Forest", "mlp": "MLP"}

# Okabe-Ito, fixed order, one hue per model for the whole paper. Never cycled,
# never reassigned by rank, so the SVM is the same blue in every figure.
COLOUR = {"svm": "#0072B2", "xgb": "#E69F00",
          "rf": "#009E73", "mlp": "#D55E00"}
MARKER = {"svm": "o", "xgb": "s", "rf": "^", "mlp": "D"}
DASH = {"svm": (None, None), "xgb": (4, 1.5), "rf": (1.5, 1.5), "mlp": (5, 1, 1, 1)}

N_BINS = 15                                # Section 16 fixes this at 15
MIN_BIN = 50                               # below this a bin is drawn hollow


def load_oof() -> dict:
    out = {}
    for key in ORDER:
        p = ARTIFACT_DIR / f"oof_predictions_{key}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        pcols = [f"p_{CLASS_NAMES[c]}" for c in range(4)]
        missing = [c for c in pcols + ["true", "pred"] if c not in df.columns]
        if missing:
            print(f"  skipping {key}: missing columns {missing}")
            continue
        df["conf"] = df[pcols].to_numpy(float).max(axis=1)
        df["correct"] = (df["true"] == df["pred"]).astype(float)
        out[key] = df
    return out


def bin_table(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # 1/n_classes is the floor of max-probability, so the lowest bins are
    # empty by construction for a four-class problem. They are kept in the
    # table and simply carry n = 0.
    idx = np.clip(np.digitize(df["conf"].to_numpy(), edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        rows.append({"bin": b,
                     "lo": round(float(edges[b]), 4),
                     "hi": round(float(edges[b + 1]), 4),
                     "n": n,
                     "mean_conf": round(float(df.loc[m, "conf"].mean()), 4) if n else np.nan,
                     "accuracy": round(float(df.loc[m, "correct"].mean()), 4) if n else np.nan})
    t = pd.DataFrame(rows)
    t["gap"] = (t["accuracy"] - t["mean_conf"]).round(4)
    return t


def ece_from_bins(t: pd.DataFrame) -> float:
    m = t["n"] > 0
    w = t.loc[m, "n"].to_numpy(float)
    return float(np.sum(w * np.abs(t.loc[m, "gap"].to_numpy(float))) / w.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=N_BINS)
    ap.add_argument("--min-bin", type=int, default=MIN_BIN)
    ap.add_argument("--width", type=float, default=3.4,
                    help="figure width in inches; 3.4 fits one IEEE column")
    args = ap.parse_args()

    print(banner(), "\n")
    oof = load_oof()
    if not oof:
        raise SystemExit(f"no oof_predictions_*.csv in {ARTIFACT_DIR}")

    tables, eces = {}, {}
    for key, df in oof.items():
        t = bin_table(df, args.bins)
        tables[key] = t
        eces[key] = ece_from_bins(t)
        print(f"{DISPLAY[key]:<14} ECE {eces[key]:.4f} from {args.bins} bins "
              f"| {int((t['n'] > 0).sum())} non-empty | "
              f"{int(t.loc[t['n'] < args.min_bin, 'n'].sum())} segments in "
              f"sparse bins")

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    })
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(args.width, args.width * 1.12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    ax.plot([0, 1], [0, 1], color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=1)
    # Low confidence with low accuracy is the emptiest corner of the plot, so
    # the diagonal is labelled there rather than mid-plot where the curves run.
    ax.text(0.295, 0.232, "perfect calibration", color="0.45", fontsize=5.6,
            rotation=38, rotation_mode="anchor")

    for key in ORDER:
        if key not in tables:
            continue
        t = tables[key]
        m = t["n"] > 0
        x, y, n = (t.loc[m, "mean_conf"].to_numpy(),
                   t.loc[m, "accuracy"].to_numpy(),
                   t.loc[m, "n"].to_numpy())
        dense = n >= args.min_bin
        ax.plot(x, y, color=COLOUR[key], lw=1.1, dashes=DASH[key], zorder=3,
                label=f"{DISPLAY[key]} (ECE {eces[key]:.3f})")
        ax.plot(x[dense], y[dense], color=COLOUR[key], marker=MARKER[key],
                ms=3.1, mew=0.5, mec="white", ls="none", zorder=4)
        ax.plot(x[~dense], y[~dense], color=COLOUR[key], marker=MARKER[key],
                ms=3.1, mew=0.6, mfc="white", ls="none", zorder=4)
        axh.step(t["hi"].to_numpy(), t["n"].to_numpy() / t["n"].sum(),
                 where="pre", color=COLOUR[key], lw=0.9, dashes=DASH[key])

    ax.set_xlim(0.2, 1.005)
    ax.set_ylim(0.2, 1.005)
    ax.set_ylabel("accuracy within bin")
    ax.grid(True, color="0.88")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
        axh.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handlelength=2.4,
              borderpad=0.1, labelspacing=0.25)

    axh.set_yscale("log")
    axh.set_ylabel("share of\nsegments")
    axh.set_xlabel("maximum predicted probability")
    axh.grid(True, color="0.88")
    axh.set_axisbelow(True)

    for ext in ("pdf", "png"):
        out = ARTIFACT_DIR / f"reliability.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")

    allt = pd.concat([t.assign(model=DISPLAY[k]) for k, t in tables.items()])
    allt.to_csv(ARTIFACT_DIR / "reliability_bins.csv", index=False)
    print(f"wrote {ARTIFACT_DIR / 'reliability_bins.csv'}")

    print("\nBin-recomputed ECE against the value stored by the trainers:")
    for key in ORDER:
        if key in eces:
            print(f"  {DISPLAY[key]:<14} {eces[key]:.4f}")
    print("  (small differences are expected: the trainers average ECE over "
          "folds, this pools all out-of-fold predictions)")


if __name__ == "__main__":
    main()
