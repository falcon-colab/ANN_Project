"""
src/s1_physics/train_classical.py

The S1 classical baselines under the full protocol: RBF-SVM, Random Forest and
XGBoost, nested 5 outer x 5 inner, measurement-grouped folds, no shortcuts.

Nothing is skipped. Every configuration is evaluated on every inner fold of
every outer fold, and every outer fold gets a final model fitted on its full
training portion. Fit counts:

    RBF-SVM         9 configs x 5 inner x 5 outer  = 225  + 5 final = 230
    Random Forest   6 configs x 5 inner x 5 outer  = 150  + 5 final = 155
    XGBoost         8 configs x 5 inner x 5 outer  = 200  + 5 final = 205

Two things make this finish in reasonable time without touching the protocol.

First, probability=False during inner selection. Section 10 selects on mean
inner-fold macro-F1, which needs predict(), not predict_proba(). Section 16
requires probabilistic analysis of the FINAL classifier only. SVC with
probability=True runs an internal 5-fold Platt calibration, so enabling it
during selection multiplies the inner loop by five for no benefit.

Second, joblib across fits. SVC is single-threaded and the 225 inner fits are
independent, so this is near-linear in cores. Model-level n_jobs is forced to
1 for the tree models so the two levels of parallelism do not oversubscribe.

Leakage control: the feature preprocessor is fitted on the current training
portion only, inside every loop. Outer-test data never touches selection,
scaling or calibration.

Run from the project root:
    python src/s1_physics/train_classical.py --probe
    python src/s1_physics/train_classical.py --models svm rf xgb --n-jobs 16
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (ParameterGrid,
                                     StratifiedGroupKFold)
from sklearn.svm import SVC

try:
    import psutil
    _PROC = psutil.Process()
except ImportError:                                   # pip install psutil
    psutil = None
    _PROC = None

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
sys.path.append(str(Path(__file__).resolve().parent))
from config import ARTIFACT_DIR, N_INNER, N_OUTER, OUTER_FOLD_SEEDS, SEED, banner
from feature_transforms import (COMPACT_SUBSET, FEATURE_NAMES,
                                FeaturePreprocessor)
from metrics import evaluate_classification

CLASS_NAMES = {0: "drone", 1: "bird", 2: "human", 3: "reflector"}

# Protocol section 18 latency protocol: batch size 1, 100 warm-up inferences,
# 1000 timed inferences, data loading excluded.
LATENCY_WARMUP = 100
LATENCY_TIMED = 1000
# Dr Farrukh, 4 Sep: "measuring latency would be tricky as the same setup can
# behave differently the second time. One should repeat experiments a number
# of times and get an average." The protocol's 100 warm-up plus 1000 timed
# inferences is therefore repeated as a whole, and the spread across
# repetitions is reported alongside the mean so run-to-run variation on the
# shared Atlas node is visible rather than hidden.
LATENCY_REPEATS = 3


def peak_memory_mb() -> float:
    """Peak resident set size in MB, or NaN if unavailable."""
    if _PROC is not None:
        info = _PROC.memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / 1e6
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1e3
    except Exception:
        return float("nan")


def measure_latency(est, X: np.ndarray, repeats: int = LATENCY_REPEATS) -> dict:
    """
    Batch-one inference latency, section 18.

    predict_proba is timed rather than predict, because the probabilistic
    outputs are what section 16 requires from the final classifier, so that is
    the operation a deployment would actually perform. The single row is
    materialised once so data loading is excluded from the timing.
    """
    row = np.ascontiguousarray(X[:1])
    means, p95s = [], []
    for _ in range(max(1, repeats)):
        for _ in range(LATENCY_WARMUP):
            est.predict_proba(row)
        t = np.empty(LATENCY_TIMED)
        for i in range(LATENCY_TIMED):
            t0 = time.perf_counter()
            est.predict_proba(row)
            t[i] = time.perf_counter() - t0
        means.append(float(t.mean() * 1e3))
        p95s.append(float(np.percentile(t, 95) * 1e3))
    return {"mean_latency_ms": float(np.mean(means)),
            "p95_latency_ms": float(np.mean(p95s)),
            "latency_repeats": len(means),
            "latency_mean_spread_ms": float(np.max(means) - np.min(means)),
            "latency_per_repeat_ms": [round(m, 4) for m in means]}


def fmt_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"


# --------------------------------------------------------------------------
# Model definitions. Grids are verbatim from protocol section 10.
# --------------------------------------------------------------------------

@dataclass
class ModelSpec:
    key: str
    name: str
    grid: dict
    needs_scaling: bool
    # SVC is the only estimator whose probability output costs anything, so
    # it is the only one that distinguishes selection from final fitting.
    prob_costs_extra: bool = False
    extra: dict = field(default_factory=dict)

    def n_configs(self) -> int:
        return len(list(ParameterGrid(self.grid)))


def build_specs() -> dict[str, ModelSpec]:
    specs = {
        "svm": ModelSpec(
            key="svm",
            name="RBF-SVM",
            grid={"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]},
            needs_scaling=True,
            prob_costs_extra=True,
            extra={"kernel": "rbf", "cache_size": 1000, "random_state": SEED},
        ),
        "rf": ModelSpec(
            key="rf",
            name="Random Forest",
            grid={"max_depth": [None, 10, 20], "min_samples_leaf": [1, 3]},
            needs_scaling=False,
            extra={"n_estimators": 500, "n_jobs": 1, "random_state": SEED},
        ),
    }
    try:
        from xgboost import XGBClassifier  # noqa: F401
        specs["xgb"] = ModelSpec(
            key="xgb",
            name="XGBoost",
            grid={"n_estimators": [200, 500], "max_depth": [3, 6],
                  "learning_rate": [0.03, 0.1]},
            needs_scaling=False,
            extra={"n_jobs": 1, "tree_method": "hist", "random_state": SEED,
                   "num_class": 4, "objective": "multi:softprob"},
        )
    except ImportError:
        print("WARNING: xgboost not installed; skipping that model")
    return specs


def make_calibrated_svm(spec, params, seed, X, y, groups, n_splits=5):
    """
    Probability-calibrated SVC whose calibration folds respect measurement
    boundaries.

    SVC(probability=True) is deprecated in scikit-learn 1.9, but the more
    important problem is that its internal Platt calibration uses a plain
    5-fold split. Segments from one recording are highly correlated, so a
    non-grouped split lets the calibrator see near-duplicates of its own
    validation data and learn optimistically sharp confidences. That biases
    ECE and Brier, which are protocol metrics and the basis of RQ9.

    Passing explicit group-aware splits fixes both at the same cost: six base
    fits either way. Everything here happens inside the outer training
    portion, so the outer test fold is never touched.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    splits = list(sgkf.split(X, y, groups))
    base = make_estimator(spec, params, seed, probability=False)
    return CalibratedClassifierCV(base, method="sigmoid", ensemble=False,
                                  cv=splits)


def make_estimator(spec: ModelSpec, params: dict, seed: int, probability: bool):
    kw = dict(spec.extra)
    kw.update(params)
    kw["random_state"] = seed
    if spec.key == "svm":
        # Do not pass `probability` at all. scikit-learn 1.9 raises a
        # FutureWarning on the parameter itself, regardless of its value, so
        # passing False produced one warning per inner fit. Probabilities for
        # the final models come from make_calibrated_svm instead.
        if probability:
            kw["probability"] = True
        return SVC(**kw)
    if spec.key == "rf":
        return RandomForestClassifier(**kw)
    from xgboost import XGBClassifier
    kw.pop("num_class", None)          # inferred from y, and passing it warns
    return XGBClassifier(**kw)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load(exclude_edge: bool = True, feature_set: str = "full") -> tuple:
    feats = pd.read_csv(ARTIFACT_DIR / "s1_features.csv")
    folds = pd.read_csv(ARTIFACT_DIR / "fold_manifest.csv")

    keep = ["segment_uid", "measurement_id", "class_id", "class_name",
            "edge_flag", "outer_fold"] + [f"inner_fold_for_outer_{k}"
                                          for k in range(N_OUTER)]
    df = feats.merge(folds[keep], on="segment_uid",
                     suffixes=("", "_fold"), validate="one_to_one")

    n_before = len(df)
    if exclude_edge:
        df = df[df["edge_flag"] == 0].reset_index(drop=True)
        print(f"Excluded {n_before - len(df)} field-of-view-edge segments "
              f"({n_before} -> {len(df)})")

    cols = list(FEATURE_NAMES if feature_set == "full" else COMPACT_SUBSET)
    return df, cols


def majority_baseline(df: pd.DataFrame) -> dict:
    """What you get by always answering the most common class."""
    out = []
    for k in range(N_OUTER):
        te = df[df["outer_fold"] == k]
        tr = df[df["outer_fold"] != k]
        maj = int(tr["class_id"].mode().iloc[0])
        pred = np.full(len(te), maj)
        out.append(f1_score(te["class_id"], pred, average="macro",
                            labels=[0, 1, 2, 3], zero_division=0))
    return {"mean_macro_f1": float(np.mean(out)),
            "std_macro_f1": float(np.std(out)),
            "per_fold": [float(v) for v in out]}


# --------------------------------------------------------------------------
# One inner evaluation: fit on inner-train, score macro-F1 on inner-val
# --------------------------------------------------------------------------

def inner_score(spec, params, tr, va, cols, seed, signed_log) -> tuple:
    pre = FeaturePreprocessor(cols, signed_log=signed_log and spec.needs_scaling)
    if spec.needs_scaling:
        Xtr, Xva = pre.fit_transform(tr), pre.transform(va)
    else:
        Xtr = tr[cols].to_numpy(float)
        Xva = va[cols].to_numpy(float)
    t0 = time.perf_counter()
    est = make_estimator(spec, params, seed, probability=False)
    est.fit(Xtr, tr["class_id"].to_numpy())
    fit_s = time.perf_counter() - t0
    pred = est.predict(Xva)
    f1 = f1_score(va["class_id"], pred, average="macro",
                  labels=[0, 1, 2, 3], zero_division=0)
    n_sv = int(getattr(est, "n_support_", np.array([0])).sum())
    return float(f1), fit_s, n_sv


def probe(spec, df, cols, signed_log) -> None:
    """
    Time one fit at the real inner size before committing to the rest.

    SVC cost grows faster than quadratically in sample count, so a single
    measured fit is worth more than any estimate. This tells you in a couple of
    minutes what the whole run will cost.
    """
    k = 0
    tr = df[(df["outer_fold"] != k) & (df[f"inner_fold_for_outer_{k}"] != 0)]
    va = df[(df["outer_fold"] != k) & (df[f"inner_fold_for_outer_{k}"] == 0)]
    def cost_rank(p):
        g = p.get("gamma", 0)
        g = 1.0 if g == "scale" else float(g) * 10   # larger gamma, more SVs
        return (p.get("C", 0) * g, p.get("n_estimators", 0),
                p.get("max_depth") or 99)
    worst = max(ParameterGrid(spec.grid), key=cost_rank)
    print(f"\n[probe] {spec.name}, worst-case config {worst}")
    print(f"[probe] inner-train {len(tr)} rows, inner-val {len(va)} rows")
    f1, fit_s, n_sv = inner_score(spec, worst, tr, va, cols, SEED, signed_log)
    n_inner = spec.n_configs() * N_INNER * N_OUTER
    print(f"[probe] one fit: {fit_s:.1f}s  macro-F1 {f1:.4f}"
          + (f"  support vectors {n_sv}" if n_sv else ""))
    print(f"[probe] {n_inner} inner fits, serial upper bound "
          f"{n_inner * fit_s / 3600:.2f} h")
    for j in (8, 16, 24):
        print(f"[probe]   at n_jobs={j}: ~{fmt_hms(n_inner * fit_s / (j*0.8))} "
              f"(assuming 80% parallel efficiency)")
    print(f"[probe] peak memory so far {peak_memory_mb():.0f} MB")


# --------------------------------------------------------------------------
# Full nested run for one model
# --------------------------------------------------------------------------

def run_model(spec, df, cols, n_jobs, signed_log, do_latency=True,
              grouped_calibration=True,
              latency_repeats=LATENCY_REPEATS) -> dict:
    print(f"\n{'='*66}\n{spec.name}: {spec.n_configs()} configs, "
          f"{spec.n_configs()*N_INNER*N_OUTER} inner fits + {N_OUTER} final\n{'='*66}")
    grid = list(ParameterGrid(spec.grid))
    fold_results, selected, timings, oof = [], [], [], []
    t_model = time.perf_counter()

    for k in range(N_OUTER):
        seed = OUTER_FOLD_SEEDS[k]
        outer_tr = df[df["outer_fold"] != k]
        outer_te = df[df["outer_fold"] == k]
        icol = f"inner_fold_for_outer_{k}"

        jobs = [
            (params, outer_tr[outer_tr[icol] != j], outer_tr[outer_tr[icol] == j])
            for params in grid for j in range(N_INNER)
        ]
        t0 = time.perf_counter()
        res = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(inner_score)(spec, p, tr, va, cols, seed, signed_log)
            for p, tr, va in jobs
        )
        inner_s = time.perf_counter() - t0
        timings.append(inner_s)

        scores = np.array([r[0] for r in res]).reshape(len(grid), N_INNER)
        mean_inner = scores.mean(axis=1)
        best_i = int(np.argmax(mean_inner))
        best = grid[best_i]
        selected.append(best)

        # Final model: full outer-training portion, probabilities enabled.
        pre = FeaturePreprocessor(cols, signed_log=signed_log and spec.needs_scaling)
        if spec.needs_scaling:
            Xtr, Xte = pre.fit_transform(outer_tr), pre.transform(outer_te)
        else:
            Xtr = outer_tr[cols].to_numpy(float)
            Xte = outer_te[cols].to_numpy(float)

        ytr = outer_tr["class_id"].to_numpy()
        t0 = time.perf_counter()
        if spec.prob_costs_extra and grouped_calibration:
            est = make_calibrated_svm(spec, best, seed, Xtr, ytr,
                                      outer_tr["measurement_id"].to_numpy())
        else:
            est = make_estimator(spec, best, seed, probability=True)
        est.fit(Xtr, ytr)
        train_s = time.perf_counter() - t0

        proba = est.predict_proba(Xte)
        oof.append(pd.DataFrame({
            "segment_uid": outer_te["segment_uid"].to_numpy(),
            "measurement_id": outer_te["measurement_id"].to_numpy(),
            "outer_fold": k,
            "true": outer_te["class_id"].to_numpy(),
            "pred": proba.argmax(1),
            **{f"p_{CLASS_NAMES[c]}": proba[:, c] for c in range(4)},
        }))
        m = evaluate_classification(proba, outer_te["class_id"].to_numpy())
        m.update(
            outer_fold=k, seed=seed, params=best,
            mean_inner_macro_f1=float(mean_inner[best_i]),
            n_train=int(len(outer_tr)), n_test=int(len(outer_te)),
            inner_seconds=round(inner_s, 1), final_fit_seconds=round(train_s, 1),
            **(measure_latency(est, Xte, latency_repeats) if do_latency else
               {"mean_latency_ms": float("nan"), "p95_latency_ms": float("nan")}),
        )
        inner_est = est
        if hasattr(est, "calibrated_classifiers_"):
            inner_est = getattr(est.calibrated_classifiers_[0], "estimator", est)
        n_sv = getattr(inner_est, "n_support_", None)
        m["n_support_vectors"] = int(n_sv.sum()) if n_sv is not None else None
        fold_results.append(m)

        model_dir = ARTIFACT_DIR / "models"
        model_dir.mkdir(exist_ok=True)
        joblib.dump({"estimator": est, "preprocessor": pre if spec.needs_scaling
                     else None, "features": cols, "params": best, "seed": seed,
                     "outer_fold": k, "class_names": CLASS_NAMES},
                    model_dir / f"{spec.key}_fold{k}.joblib")

        elapsed = time.perf_counter() - t_model
        eta = elapsed / (k + 1) * (N_OUTER - k - 1)
        print(f"  fold {k}: inner {fmt_hms(inner_s)} | final {fmt_hms(train_s)} "
              f"| best {best}")
        print(f"          inner F1 {mean_inner[best_i]:.4f} | TEST F1 "
              f"{m['macro_f1']:.4f} | latency {m['mean_latency_ms']:.3f} ms "
              f"| elapsed {fmt_hms(elapsed)}, est. remaining {fmt_hms(eta)}")

    oof_df = pd.concat(oof, ignore_index=True)
    oof_path = ARTIFACT_DIR / f"oof_predictions_{spec.key}.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"  wrote {oof_path.name} ({len(oof_df)} rows) and 5 model files")

    f1s = [r["macro_f1"] for r in fold_results]
    summary = {
        "model_name": spec.name,
        "n_configs": spec.n_configs(),
        "n_inner_fits": spec.n_configs() * N_INNER * N_OUTER,
        "outer_fold_scores": [round(v, 4) for v in f1s],
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
        "ece": float(np.mean([r["ece"] for r in fold_results])),
        "brier_score": float(np.mean([r["brier_score"] for r in fold_results])),
        "selected_params_per_fold": selected,
        "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in fold_results])),
        "p95_latency_ms": float(np.mean([r["p95_latency_ms"] for r in fold_results])),
        "latency_mean_spread_ms": float(np.mean(
            [r.get("latency_mean_spread_ms", float("nan")) for r in fold_results])),
        "inner_seconds_total": round(float(sum(timings)), 1),
        "wall_seconds": round(time.perf_counter() - t_model, 1),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "per_fold": fold_results,
    }
    print(f"  {spec.name}: macro-F1 {summary['mean_macro_f1']:.4f} "
          f"+/- {summary['std_macro_f1']:.4f}  |  total "
          f"{fmt_hms(summary['wall_seconds'])} "
          f"(inner {fmt_hms(summary['inner_seconds_total'])})")
    return summary


# --------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["svm", "rf", "xgb"])
    ap.add_argument("--features", choices=["full", "compact"], default="full")
    ap.add_argument("--no-signed-log", action="store_true")
    ap.add_argument("--keep-edge", action="store_true",
                    help="keep the 67 field-of-view-edge segments")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--probe", action="store_true",
                    help="time one worst-case fit per model, then stop")
    ap.add_argument("--latency-repeats", type=int, default=LATENCY_REPEATS,
                    help="how many times to repeat the whole 100 warm-up plus "
                         "1000 timed protocol; the spread across repetitions "
                         "is reported")
    ap.add_argument("--skip-latency", action="store_true",
                    help="skip the 100 warm-up + 1000 timed batch-one "
                         "inferences. Section 18 requires that measurement "
                         "from the agreed Atlas node type anyway, so it is "
                         "wasted locally and costs minutes per model.")
    ap.add_argument("--ungrouped-calibration", action="store_true",
                    help="use SVC(probability=True) internal Platt scaling "
                         "instead of measurement-grouped calibration folds. "
                         "For the ECE comparison only; not the default.")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t_script = time.perf_counter()
    print(banner(), "\n")
    df, cols = load(exclude_edge=not args.keep_edge, feature_set=args.features)
    signed_log = not args.no_signed_log
    print(f"{len(df)} segments | {len(cols)} features ({args.features}) | "
          f"signed-log {'on' if signed_log else 'off'} | n_jobs {args.n_jobs} | "
          f"latency {'skipped' if args.skip_latency else 'measured'}")
    print("class balance: "
          + ", ".join(f"{n} {c/len(df):.1%}"
                      for n, c in df["class_name"].value_counts().items()))

    specs = build_specs()
    chosen = [specs[m] for m in args.models if m in specs]

    if args.probe:
        for spec in chosen:
            probe(spec, df, cols, signed_log)
        print("\nProbe only. Re-run without --probe to execute the full protocol.")
        return

    base = majority_baseline(df)
    print(f"\nMajority-class baseline macro-F1: {base['mean_macro_f1']:.4f} "
          f"+/- {base['std_macro_f1']:.4f}  (beat this or report nothing)")

    results = {"latency_measured": not args.skip_latency,
               "grouped_calibration": not args.ungrouped_calibration,
               "generated": datetime.now(timezone.utc).isoformat(),
               "commit": git_commit(), "host": platform.node(),
               "feature_set": args.features, "features": cols,
               "signed_log": signed_log, "edge_excluded": not args.keep_edge,
               "n_segments": int(len(df)), "seed": SEED,
               "outer_fold_seeds": list(OUTER_FOLD_SEEDS),
               "majority_baseline": base, "models": {}}

    for spec in chosen:
        results["models"][spec.key] = run_model(
            spec, df, cols, args.n_jobs, signed_log,
            do_latency=not args.skip_latency,
            grouped_calibration=not args.ungrouped_calibration,
            latency_repeats=args.latency_repeats)

    tag = args.tag or args.features
    out = ARTIFACT_DIR / f"results_classical_{tag}.json"
    out.write_text(json.dumps(results, indent=2, default=str))

    ledger = ARTIFACT_DIR / "run_ledger.csv"
    rows = [{
        "timestamp": results["generated"], "atlas_job_id":
            __import__("os").getenv("SLURM_JOB_ID", "local"),
        "student_track": "S1", "model_name": r["model_name"],
        "commit_hash": results["commit"], "seed": SEED,
        "config_json": json.dumps(r["selected_params_per_fold"]),
        "status": "complete", "val_macro_f1": "",
        "test_macro_f1": round(r["mean_macro_f1"], 4),
        "test_accuracy": round(r["accuracy"], 4),
        "ece": round(r["ece"], 4), "brier_score": round(r["brier_score"], 4),
        "training_time_sec": r["wall_seconds"],
        "peak_memory_mb": r["peak_memory_mb"],
        "mean_latency_ms": round(r["mean_latency_ms"], 4),
        "p95_latency_ms": round(r["p95_latency_ms"], 4),
    } for r in results["models"].values()]
    new = pd.DataFrame(rows)
    if ledger.exists():
        new = pd.concat([pd.read_csv(ledger), new], ignore_index=True)
    new.to_csv(ledger, index=False)

    total_s = time.perf_counter() - t_script
    results["total_wall_seconds"] = round(total_s, 1)
    results["peak_memory_mb"] = round(peak_memory_mb(), 1)
    out.write_text(json.dumps(results, indent=2, default=str))

    hdr = (f"{'model':<16}{'macro-F1':>10}{'std':>8}{'acc':>8}{'ECE':>8}"
           f"{'Brier':>8}{'lat ms':>9}{'p95 ms':>9}{'time':>12}")
    print("\n" + hdr)
    print("-" * len(hdr))
    print(f"{'majority':<16}{base['mean_macro_f1']:>10.4f}"
          f"{base['std_macro_f1']:>8.4f}{'-':>8}{'-':>8}{'-':>8}{'-':>9}{'-':>9}{'-':>12}")
    for r in results["models"].values():
        print(f"{r['model_name']:<16}{r['mean_macro_f1']:>10.4f}"
              f"{r['std_macro_f1']:>8.4f}{r['accuracy']:>8.4f}"
              f"{r['ece']:>8.4f}{r['brier_score']:>8.4f}"
              f"{r['mean_latency_ms']:>9.3f}{r['p95_latency_ms']:>9.3f}"
              f"{fmt_hms(r['wall_seconds']):>12}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<16}{'':>10}{'':>8}{'':>8}{'':>8}{'':>8}{'':>9}{'':>9}"
          f"{fmt_hms(total_s):>12}")
    print(f"\nPeak memory {results['peak_memory_mb']:.0f} MB"
          f" | host {results['host']} | commit {results['commit']}")
    print(f"Wrote {out.name} and run_ledger.csv")
    print("\nNote: latency here is from this machine. Section 18 requires the "
          "figure that goes in the paper to come from the agreed Atlas CPU "
          "node type, so re-run there before the results freeze.")


if __name__ == "__main__":
    main()
