"""
src/s1_physics/run_ablations.py

The five remaining S1 analyses from protocol section 10, in one pass:

    compact five-feature subset
    feature-family ablation        (family-only and leave-one-family-out)
    individual-feature ablation    (leave-one-out over all ten)
    permutation importance
    data-efficiency experiment     (20, 40, 60, 80, 100% of training data)

Hyperparameters are NOT re-tuned per condition. Each ablation reuses the
configuration selected for that outer fold in the full-feature run, which is
read from results_classical_full.json and results_mlp.json. Re-running inner
selection for every one of these conditions would multiply the fit count by
25 and would also violate the 12-configuration tuning cap in spirit, since
the cap is per principal model rather than per ablation. Holding the
configuration fixed is also the cleaner comparison: it isolates the effect of
the features from the effect of retuning.

Everything else follows the frozen protocol: the same measurement-grouped
outer folds, the same per-fold standardisation fitted on training data only,
the same outer-fold seeds.

Data efficiency subsamples MEASUREMENTS, not segments. Dropping random
segments would leave every recording partially present and would make the
low-data regime far easier than it really is, because a near-duplicate of
almost every training segment would still be there.

Run from the project root:
    python src/s1_physics/run_ablations.py --list
    python src/s1_physics/run_ablations.py --models svm xgb --n-jobs 16
    python src/s1_physics/run_ablations.py --only data_efficiency
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
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
sys.path.append(str(Path(__file__).resolve().parent))
from config import ARTIFACT_DIR, N_OUTER, OUTER_FOLD_SEEDS, banner
from feature_transforms import (COMPACT_SUBSET_CORRELATION,
                                COMPACT_SUBSET_IMPORTANCE, FEATURE_NAMES,
                                FeaturePreprocessor)
from train_classical import build_specs, load, make_estimator

# Families follow the physical quantity each descriptor measures, not the
# order they appear in the protocol document.
FAMILIES = {
    "energy_distribution": ("zero_doppler_ratio", "side_lobe_energy_ratio",
                            "side_lobe_entropy"),
    "spectral_shape": ("spectral_entropy", "doppler_spread",
                       "doppler_bandwidth_80", "skewness", "kurtosis"),
    "temporal": ("temporal_energy_variance", "temporal_entropy"),
}

DATA_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def selected_params() -> dict[str, list[dict]]:
    """The configuration chosen per outer fold in the full-feature run."""
    out = {}
    p = ARTIFACT_DIR / "results_classical_full.json"
    if p.exists():
        for k, m in json.loads(p.read_text())["models"].items():
            if "selected_params_per_fold" not in m:
                print(f"  warning: {k} has no selected_params_per_fold; "
                      f"re-run train_classical.py to regenerate the results "
                      f"file before ablating")
                continue
            out[k] = m["selected_params_per_fold"]
    p = ARTIFACT_DIR / "results_mlp.json"
    if p.exists():
        m = json.loads(p.read_text())["model"]
        if "selected_params_per_fold" in m:
            out["mlp"] = m["selected_params_per_fold"]
    if not out:
        raise SystemExit("no selected hyperparameters found; run "
                         "train_classical.py first")
    return out


def subsample_measurements(df: pd.DataFrame, fraction: float,
                           seed: int) -> pd.DataFrame:
    """
    Keep a stratified fraction of MEASUREMENTS, then every segment of those.

    Sampling segments instead would leave a near-duplicate of almost every
    dropped segment in the training set, so the low-data regime would not
    actually be low-data.
    """
    if fraction >= 1.0:
        return df
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby("class_id", sort=True):
        ms = g["measurement_id"].unique()
        n = max(2, int(round(len(ms) * fraction)))
        keep.extend(rng.choice(ms, size=min(n, len(ms)), replace=False))
    return df[df["measurement_id"].isin(set(keep))]


def one_fold(spec, params, df, cols, k, seed, fraction, signed_log):
    tr = df[df["outer_fold"] != k]
    te = df[df["outer_fold"] == k]
    tr = subsample_measurements(tr, fraction, seed)
    if tr["class_id"].nunique() < 4:
        return float("nan"), 0
    pre = FeaturePreprocessor(cols, signed_log=signed_log and spec.needs_scaling)
    if spec.needs_scaling:
        Xtr, Xte = pre.fit_transform(tr), pre.transform(te)
    else:
        Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
    est = make_estimator(spec, params, seed, probability=False)
    est.fit(Xtr, tr["class_id"].to_numpy())
    f1 = f1_score(te["class_id"], est.predict(Xte), average="macro",
                  labels=[0, 1, 2, 3], zero_division=0)
    return float(f1), int(len(tr))


def run_condition(spec, df, cols, params_per_fold, signed_log, fraction=1.0):
    scores, sizes = [], []
    for k in range(N_OUTER):
        seed = OUTER_FOLD_SEEDS[k]
        p = params_per_fold[k] if k < len(params_per_fold) else params_per_fold[0]
        f1, n = one_fold(spec, p, df, cols, k, seed, fraction, signed_log)
        scores.append(f1)
        sizes.append(n)
    return {"mean": round(float(np.nanmean(scores)), 4),
            "std": round(float(np.nanstd(scores)), 4),
            "per_fold": [round(s, 4) for s in scores],
            "mean_train_rows": int(np.mean(sizes))}


def conditions_for(cols_full) -> list[tuple[str, str, list[str]]]:
    out = [("baseline", "all ten descriptors", list(cols_full)),
           ("compact5_correlation",
            "five features chosen from the correlation structure, before "
            "any model was trained",
            list(COMPACT_SUBSET_CORRELATION)),
           ("compact5_importance",
            "five features chosen from permutation importance on the "
            "full-feature models, so not a blind choice",
            list(COMPACT_SUBSET_IMPORTANCE))]
    for fam, feats in FAMILIES.items():
        out.append((f"only_{fam}", f"{fam} family alone", list(feats)))
        rest = [c for c in cols_full if c not in feats]
        out.append((f"drop_{fam}", f"all but the {fam} family", rest))
    for f in cols_full:
        out.append((f"drop_{f}", f"leave out {f}",
                    [c for c in cols_full if c != f]))
    return out


def permutation_importance_saved(spec_key: str, df, cols, n_repeats=5,
                                 seed=2026) -> pd.DataFrame:
    """
    Permutation importance from the SAVED final models, so nothing is refitted.

    Each feature is shuffled within the outer test fold and the drop in
    macro-F1 recorded. Shuffling breaks the feature's relationship with the
    label while leaving its marginal distribution intact.
    """
    import joblib
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(N_OUTER):
        path = ARTIFACT_DIR / "models" / f"{spec_key}_fold{k}.joblib"
        if not path.exists():
            continue
        blob = joblib.load(path)
        est, pre = blob["estimator"], blob["preprocessor"]
        te = df[df["outer_fold"] == k]
        X = pre.transform(te) if pre is not None else te[cols].to_numpy(float)
        y = te["class_id"].to_numpy()
        base = f1_score(y, est.predict(X), average="macro",
                        labels=[0, 1, 2, 3], zero_division=0)
        for j, name in enumerate(cols):
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                drops.append(base - f1_score(y, est.predict(Xp),
                                             average="macro",
                                             labels=[0, 1, 2, 3],
                                             zero_division=0))
            rows.append({"outer_fold": k, "feature": name,
                         "importance": float(np.mean(drops))})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    return (d.groupby("feature")["importance"]
             .agg(["mean", "std"]).round(4)
             .sort_values("mean", ascending=False).reset_index())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["svm", "xgb", "rf"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--no-signed-log", action="store_true")
    ap.add_argument("--keep-edge", action="store_true")
    ap.add_argument("--only", choices=["ablations", "data_efficiency",
                                       "permutation"], default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    print(banner(), "\n")
    df, cols = load(exclude_edge=not args.keep_edge, feature_set="full")
    signed_log = not args.no_signed_log
    conds = conditions_for(cols)

    if args.list:
        print(f"{len(conds)} ablation conditions x {N_OUTER} folds "
              f"x {len(args.models)} models = "
              f"{len(conds) * N_OUTER * len(args.models)} fits")
        for name, desc, c in conds:
            print(f"  {name:<28} {len(c):>2} features  {desc}")
        print(f"\nplus data efficiency: {len(DATA_FRACTIONS)} fractions "
              f"x {N_OUTER} folds x {len(args.models)} models = "
              f"{len(DATA_FRACTIONS) * N_OUTER * len(args.models)} fits")
        return

    specs = build_specs()
    params = selected_params()
    results = {"generated": datetime.now(timezone.utc).isoformat(),
               "signed_log": signed_log, "families": {k: list(v) for k, v
                                                      in FAMILIES.items()},
               "compact_subset_correlation": list(COMPACT_SUBSET_CORRELATION),
               "compact_subset_importance": list(COMPACT_SUBSET_IMPORTANCE),
               "ablations": {}, "data_efficiency": {}, "permutation": {}}
    t0 = time.perf_counter()

    for key in args.models:
        if key not in specs or key not in params:
            print(f"skipping {key}: no saved selection")
            continue
        spec, pf = specs[key], params[key]

        if args.only in (None, "ablations"):
            print(f"=== {spec.name}: {len(conds)} ablation conditions ===")
            out = Parallel(n_jobs=args.n_jobs)(
                delayed(run_condition)(spec, df, c, pf, signed_log)
                for _, _, c in conds)
            base = out[0]["mean"]
            results["ablations"][key] = {}
            for (name, desc, c), r in zip(conds, out):
                r.update(description=desc, n_features=len(c),
                         delta=round(r["mean"] - base, 4))
                results["ablations"][key][name] = r
            rows = sorted(results["ablations"][key].items(),
                          key=lambda kv: kv[1]["delta"])
            print(f"  {'condition':<28}{'feats':>6}{'macro-F1':>10}{'delta':>9}")
            for name, r in rows:
                print(f"  {name:<28}{r['n_features']:>6}{r['mean']:>10.4f}"
                      f"{r['delta']:>+9.4f}")
            print()

        if args.only in (None, "data_efficiency"):
            print(f"=== {spec.name}: data efficiency ===")
            out = Parallel(n_jobs=args.n_jobs)(
                delayed(run_condition)(spec, df, cols, pf, signed_log, f)
                for f in DATA_FRACTIONS)
            results["data_efficiency"][key] = {}
            print(f"  {'fraction':>9}{'train rows':>12}{'macro-F1':>10}{'std':>8}")
            for f, r in zip(DATA_FRACTIONS, out):
                results["data_efficiency"][key][f"{f:.1f}"] = r
                print(f"  {f:>9.0%}{r['mean_train_rows']:>12}"
                      f"{r['mean']:>10.4f}{r['std']:>8.4f}")
            print()

        if args.only in (None, "permutation"):
            imp = permutation_importance_saved(key, df, cols)
            if len(imp):
                results["permutation"][key] = imp.to_dict("records")
                print(f"=== {spec.name}: permutation importance ===")
                print(imp.to_string(index=False))
                print()
            else:
                print(f"  no saved {key} models; skipping permutation "
                      f"importance\n")

    out = ARTIFACT_DIR / "results_ablations.json"

    # --only computes ONE section, and this file holds all three. Without the
    # merge below, running --only permutation after a full run would write a
    # file whose ablations and data_efficiency sections are empty, silently
    # destroying an hour of compute that nothing would flag afterwards.
    if args.only and out.exists():
        try:
            prev = json.loads(out.read_text())
        except json.JSONDecodeError:
            prev = {}
            print("  warning: existing results file is unreadable; not merging")
        for section in ("ablations", "data_efficiency", "permutation"):
            if section != args.only and prev.get(section):
                results[section] = prev[section]
                print(f"  kept {section} from the previous run")
        results["partial_run"] = args.only

    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"total {(time.perf_counter() - t0) / 60:.1f} min | wrote {out.name}")


if __name__ == "__main__":
    main()
