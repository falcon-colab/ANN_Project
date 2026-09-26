"""
src/s1_physics/mlp_data_efficiency.py

The MLP's data-efficiency curve, the last gap in the S1 analysis set.

run_ablations.py covers the three classical models because they are scikit-learn
estimators it can refit in a loop. The MLP needs a PyTorch training run per
point, so it lives here instead of being bolted into that script.

Protocol, matched to run_ablations.py so the curves are comparable:
  * subsample MEASUREMENTS, never segments. Dropping segments would leave a
    near-duplicate of almost every dropped one in the training set, so the
    low-data regime would not actually be low data;
  * the same five fractions, 20 to 100 per cent, and the same per-class
    stratification with at least two measurements per class;
  * the outer test fold is never touched. Only the training portion shrinks;
  * hyperparameters are NOT re-tuned. Each fold reuses the configuration the
    full-data nested run selected for that fold, read from results_mlp.json,
    exactly as the classical ablations reuse theirs. Re-tuning at every
    fraction would measure tuning effort rather than data efficiency;
  * early stopping keeps its own inner-fold-0 split from the reduced training
    portion, so no test data influences when training ends.

Cost: five fractions times five folds is 25 trainings, a few minutes on a GPU
and roughly half an hour on CPU.

    python src/s1_physics/mlp_data_efficiency.py
    python src/s1_physics/mlp_data_efficiency.py --fractions 0.2 0.6 1.0

Writes results_mlp_data_efficiency.json and prints a pgfplots coordinate line
ready to paste into the paper's figure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
sys.path.append(str(Path(__file__).resolve().parent))

from config import ARTIFACT_DIR, N_OUTER, OUTER_FOLD_SEEDS, banner  # noqa: E402
from feature_transforms import FeaturePreprocessor                  # noqa: E402
from train_mlp import (evaluate, load, resolve_device, set_seed,     # noqa: E402
                       to_tensors, train_one)

FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
FALLBACK_PARAMS = {"lr": 1e-3, "weight_decay": 1e-5}


def selected_params_per_fold() -> list[dict]:
    """The configuration the full-data run chose for each outer fold."""
    p = ARTIFACT_DIR / "results_mlp.json"
    if not p.exists():
        print(f"  warning: {p.name} not found; falling back to "
              f"{FALLBACK_PARAMS} for every fold, which is NOT the same as "
              f"reusing the selected configuration")
        return [dict(FALLBACK_PARAMS)] * N_OUTER
    sel = json.loads(p.read_text())["model"].get("selected_params_per_fold")
    if not sel:
        print("  warning: no selected_params_per_fold in results_mlp.json")
        return [dict(FALLBACK_PARAMS)] * N_OUTER
    return [dict(s) for s in sel]


def subsample_measurements(df: pd.DataFrame, fraction: float,
                           seed: int) -> pd.DataFrame:
    """Identical rule to run_ablations.subsample_measurements."""
    if fraction >= 1.0:
        return df
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby("class_id", sort=True):
        ms = g["measurement_id"].unique()
        n = max(2, int(round(len(ms) * fraction)))
        keep.extend(rng.choice(ms, size=min(n, len(ms)), replace=False))
    return df[df["measurement_id"].isin(set(keep))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--features", choices=["full", "compact"], default="full")
    ap.add_argument("--keep-edge", action="store_true")
    args = ap.parse_args()

    print(banner(), "\n")
    device = resolve_device(args.device)
    print(f"device {device}")
    df, cols = load(exclude_edge=not args.keep_edge, feature_set=args.features)
    params = selected_params_per_fold()

    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "device": str(device),
           "features": cols,
           "fractions": args.fractions,
           "selected_params_per_fold": params,
           "note": ("measurement-level subsampling of the training portion "
                    "only; hyperparameters reused from the full-data nested "
                    "run, not re-tuned"),
           "data_efficiency": {}}
    t_all = time.perf_counter()

    for frac in args.fractions:
        scores, rows = [], []
        for k in range(N_OUTER):
            seed = OUTER_FOLD_SEEDS[k]
            set_seed(seed)
            outer_tr = df[df["outer_fold"] != k]
            outer_te = df[df["outer_fold"] == k]
            icol = f"inner_fold_for_outer_{k}"

            sub = subsample_measurements(outer_tr, frac, seed)
            fit_tr = sub[sub[icol] != 0]
            stop_va = sub[sub[icol] == 0]
            if len(stop_va) < 100 or fit_tr["class_id"].nunique() < 4:
                # A fraction so small that a class or the early-stopping split
                # vanishes would report a meaningless number rather than a
                # hard low-data result, so it is skipped and recorded as such.
                print(f"  fraction {frac:.1f} fold {k}: skipped "
                      f"({len(fit_tr)} train, {len(stop_va)} stop rows, "
                      f"{fit_tr['class_id'].nunique()} classes)")
                continue

            pre = FeaturePreprocessor(cols)
            Xtr, ytr = to_tensors(pre, fit_tr, fit_tr, device, fit=True)
            Xva, yva = to_tensors(pre, fit_tr, stop_va, device)
            Xte, yte = to_tensors(pre, fit_tr, outer_te, device)

            t0 = time.perf_counter()
            _, _, model, best_ep = train_one(Xtr, ytr, Xva, yva, params[k],
                                             seed, device)
            # evaluate() returns (loss, macro_f1, proba), in that order.
            # Unpacking it as (f1, _, _) silently records the loss instead,
            # which produces a curve that FALLS as data grows because the loss
            # is improving. The sanity check at the end of main() exists to
            # catch that class of mistake rather than trusting this line.
            _, f1, _ = evaluate(model, Xte, yte)
            scores.append(float(f1))
            rows.append({"outer_fold": k, "macro_f1": round(float(f1), 4),
                         "n_train_rows": int(len(fit_tr)),
                         "n_train_measurements":
                             int(sub["measurement_id"].nunique()),
                         "best_epoch": int(best_ep),
                         "seconds": round(time.perf_counter() - t0, 1)})
            print(f"  fraction {frac:.1f} fold {k}: macro-F1 {f1:.4f} "
                  f"| {len(fit_tr)} rows from "
                  f"{sub['measurement_id'].nunique()} measurements "
                  f"| best epoch {best_ep} "
                  f"| {time.perf_counter() - t0:.0f}s")

        if scores:
            out["data_efficiency"][f"{frac}"] = {
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
                "per_fold": [round(s, 4) for s in scores],
                "mean_train_rows": int(np.mean([r["n_train_rows"] for r in rows])),
                "folds": rows}
            d = out["data_efficiency"][f"{frac}"]
            print(f"  -> {frac:.1f}: {d['mean']:.4f} +/- {d['std']:.4f}\n")

    # Sanity check against the full-data nested run. At fraction 1.0 this
    # script trains on the same rows as train_mlp.py's final fit, so the two
    # must agree to within run-to-run variation. A large gap means the wrong
    # quantity is being recorded, not that data efficiency is surprising.
    full = out["data_efficiency"].get("1.0")
    ref_path = ARTIFACT_DIR / "results_mlp.json"
    if full and ref_path.exists():
        ref = json.loads(ref_path.read_text())["model"]["mean_macro_f1"]
        gap = abs(full["mean"] - ref)
        verdict = "OK" if gap <= 0.05 else "SUSPECT"
        print(f"\nsanity check [{verdict}]: fraction 1.0 gives "
              f"{full['mean']:.4f} against {ref:.4f} from the full nested run "
              f"(gap {gap:.4f})")
        if gap > 0.05:
            print("  The curve should end where the full run ended. Check that "
                  "the recorded quantity is macro-F1 and not the loss, and "
                  "that the training rows match.")

    path = ARTIFACT_DIR / "results_mlp_data_efficiency.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"total {(time.perf_counter() - t_all)/60:.1f} min | wrote {path.name}\n")

    pts = " ".join(f"({int(float(f)*100)},{d['mean']:.4f})"
                   for f, d in out["data_efficiency"].items())
    print("pgfplots line for the paper figure:")
    print(f"  \\addplot[mark=diamond*, mark size=1.4pt, red!70!black] "
          f"coordinates {{{pts}}};")


if __name__ == "__main__":
    main()
