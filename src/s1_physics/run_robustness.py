"""
src/s1_physics/run_robustness.py

Protocol section 17, plus the two S1 analyses that depend on it: feature
stability under corruption, and clean-versus-corrupted importance.

Nine conditions: clean, Gaussian noise at -10, -5, 0, +5 and +10 dB, Doppler
blur at 1, 2 and 4 pixels, and the combined condition of 0 dB with 2 pixel
blur.

Three rules from the protocol are enforced structurally rather than by
convention:

  corruption is applied AFTER fold splitting. Every segment belongs to exactly
  one outer test fold, so a condition corrupts each segment once and evaluates
  it with the model trained on the folds that exclude it.

  nothing is retuned or refitted on corrupted data. The saved final models are
  loaded and only ever used for inference, so no corrupted sample can leak
  into any fitting decision.

  the corrupted samples are identical for every model, because the noise seed
  is derived from segment_uid rather than from iteration order.

Noise is added to the complex slow-time signal before the transform, so these
are signal-domain results and may be described as such. Doppler blur is a
representation-domain operation and keeps the hedged wording.

A natural cross-check exists in the data: corner reflector recall already
falls at long range, where returns are weaker. If the artificial SNR curve has
the same shape as that natural degradation, the corruption is physically
meaningful rather than an arbitrary perturbation.

Run from the project root:
    python src/s1_physics/run_robustness.py --list
    python src/s1_physics/run_robustness.py --models svm xgb rf --n-jobs 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
sys.path.append(str(Path(__file__).resolve().parent))
from config import (ARTIFACT_DIR, DATA_DIR, N_DOPPLER, N_OUTER, N_RANGE,
                    banner)
from corruption import add_signal_noise, conditions, doppler_blur
from feature_extractor import FEATURE_NAMES, extract_features
from feature_transforms import FeaturePreprocessor
from preprocess_iq import linear_power, unpack_measurement

CLASS_NAMES = {0: "drone", 1: "bird", 2: "human", 3: "reflector"}


def features_for_measurement(measurement_id: str, seg_idx: np.ndarray,
                             uids: np.ndarray, cond: dict) -> np.ndarray:
    """Feature matrix for one measurement under one corruption condition."""
    iq_all = unpack_measurement(DATA_DIR / measurement_id)
    rows = np.empty((len(seg_idx), len(FEATURE_NAMES)), dtype=float)
    for i, (j, uid) in enumerate(zip(seg_idx, uids)):
        iq = iq_all[j]
        # The corrupted SIGNAL must feed both the Doppler power and the
        # in-segment STFT that the temporal descriptors use. Passing the clean
        # signal to extract_features left the temporal features untouched by
        # noise, which showed up as a Spearman correlation of exactly 1.000.
        iq_c = iq if cond["snr_db"] is None else add_signal_noise(
            iq, cond["snr_db"], uid)
        power = linear_power(iq_c)
        if cond["sigma_px"] is not None:
            power = doppler_blur(power, cond["sigma_px"])
        feats = extract_features(power, iq_c)
        rows[i] = [feats[n] for n in FEATURE_NAMES]
    return rows


def build_condition(manifest: pd.DataFrame, cond: dict, n_jobs: int) -> pd.DataFrame:
    groups = list(manifest.groupby("measurement_id", sort=False))
    out = Parallel(n_jobs=n_jobs)(
        delayed(features_for_measurement)(
            mid, g["segment_index"].to_numpy(), g["segment_uid"].to_numpy(), cond)
        for mid, g in groups)
    df = pd.concat(
        [pd.DataFrame(rows, columns=list(FEATURE_NAMES), index=g.index)
         for rows, (_, g) in zip(out, groups)]).sort_index()
    for col in ("segment_uid", "measurement_id", "class_id", "outer_fold"):
        df[col] = manifest[col].to_numpy()
    return df


def _load_predictor(model_key: str, k: int):
    """
    Return (predict, transform, cols) for fold k, or None if absent.

    Two storage formats have to be handled. The three classical models are
    joblib blobs holding an estimator plus the fold's fitted preprocessor. The
    MLP is a torch checkpoint holding a state dict plus the scaler's mean and
    scale, because a FeaturePreprocessor is not picklable across torch
    versions. Reconstructing the transform from those two arrays reproduces
    exactly what the network saw in training: signed log on the heavy-tailed
    moments, then the fold's standardisation.

    Without this, robustness covered three of the four models and the MLP's
    behaviour under corruption was simply unknown.
    """
    jb = ARTIFACT_DIR / "models" / f"{model_key}_fold{k}.joblib"
    if jb.exists():
        blob = joblib.load(jb)
        est, pre, cols = blob["estimator"], blob["preprocessor"], blob["features"]

        def transform(te, pre=pre, cols=cols):
            return pre.transform(te) if pre is not None \
                else te[cols].to_numpy(float)

        return est.predict, transform, cols

    pt = ARTIFACT_DIR / "models" / f"{model_key}_fold{k}.pt"
    if pt.exists():
        import torch                                   # only needed for the MLP
        from feature_transforms import HEAVY_TAILED, signed_log1p
        from train_mlp import MLP, resolve_device

        dev = resolve_device("auto")
        ck = torch.load(pt, map_location=dev, weights_only=False)
        cols = list(ck["features"])
        mean = np.asarray(ck["scaler_mean"], dtype=float)
        scale = np.asarray(ck["scaler_scale"], dtype=float)
        net = MLP(len(cols)).to(dev)
        net.load_state_dict(ck["state_dict"])
        net.eval()          # dropout off, batch-norm running statistics

        def transform(te, cols=cols, mean=mean, scale=scale):
            X = te[cols].to_numpy(float)
            idx = [i for i, c in enumerate(cols) if c in HEAVY_TAILED]
            if idx:
                X = X.copy()
                X[:, idx] = signed_log1p(X[:, idx])
            return (X - mean) / scale

        def predict(X, net=net, dev=dev):
            X = np.asarray(X, dtype=np.float32)
            bad = int((~np.isfinite(X)).sum())
            if bad:
                # Severe corruption can make a descriptor undefined. The trees
                # and the SVM would raise here; zeroing keeps the run alive but
                # is reported, because a silent zero is a standardised value of
                # "average" and would flatter the model.
                print(f"    note: {bad} non-finite feature values replaced "
                      f"by 0 before the MLP forward pass")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.inference_mode():
                t = torch.as_tensor(X, device=dev)
                return net(t).argmax(dim=1).cpu().numpy()

        return predict, transform, cols
    return None


def evaluate(df: pd.DataFrame, model_key: str) -> dict:
    """Score the saved per-fold models on their own outer test folds."""
    per_fold, preds, trues = [], [], []
    for k in range(N_OUTER):
        loaded = _load_predictor(model_key, k)
        if loaded is None:
            continue
        predict, transform, cols = loaded
        te = df[df["outer_fold"] == k]
        X = transform(te)
        y = te["class_id"].to_numpy()
        p = predict(X)
        per_fold.append(f1_score(y, p, average="macro", labels=[0, 1, 2, 3],
                                 zero_division=0))
        preds.append(p)
        trues.append(y)
    if not per_fold:
        return {}
    y = np.concatenate(trues)
    p = np.concatenate(preds)
    return {"mean_macro_f1": round(float(np.mean(per_fold)), 4),
            "std_macro_f1": round(float(np.std(per_fold)), 4),
            "per_fold": [round(v, 4) for v in per_fold],
            "per_class_f1": {CLASS_NAMES[c]: round(float(f1_score(
                y == c, p == c, zero_division=0)), 4) for c in range(4)}}


def feature_stability(clean: pd.DataFrame, corrupted: pd.DataFrame) -> dict:
    """
    How much each descriptor moves under corruption.

    A feature a model depends on that also shifts strongly under noise is a
    fragility; one that shifts little is a candidate for a robust subset.
    """
    out = {}
    for name in FEATURE_NAMES:
        a = clean[name].to_numpy(float)
        b = corrupted[name].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        rho = spearmanr(a[ok], b[ok]).statistic
        scale = np.median(np.abs(a[ok])) or 1.0
        out[name] = {"spearman": round(float(rho), 4),
                     "median_abs_shift": round(
                         float(np.median(np.abs(b[ok] - a[ok])) / scale), 4)}
    return out


def permutation_importance(df: pd.DataFrame, model_key: str, n_repeats=3,
                           seed=2026) -> dict:
    rng = np.random.default_rng(seed)
    acc = {n: [] for n in FEATURE_NAMES}
    for k in range(N_OUTER):
        loaded = _load_predictor(model_key, k)
        if loaded is None:
            continue
        predict, transform, cols = loaded
        te = df[df["outer_fold"] == k]
        X = transform(te)
        y = te["class_id"].to_numpy()
        base = f1_score(y, predict(X), average="macro", labels=[0, 1, 2, 3],
                        zero_division=0)
        for j, name in enumerate(cols):
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                drops.append(base - f1_score(y, predict(Xp),
                                             average="macro",
                                             labels=[0, 1, 2, 3],
                                             zero_division=0))
            acc[name].append(float(np.mean(drops)))
    return {n: round(float(np.mean(v)), 4) for n, v in acc.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["svm", "xgb", "rf"],
                    help="mlp is supported too, from its .pt checkpoints")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--keep-edge", action="store_true")
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    conds = conditions()
    if args.list:
        print(f"{len(conds)} conditions, each requiring a full feature pass "
              f"over every segment:")
        for c in conds:
            print(f"  {c['name']:<12} snr={c['snr_db']} sigma={c['sigma_px']}")
        return

    print(banner(), "\n")
    manifest = pd.read_csv(ARTIFACT_DIR / "dataset_manifest.csv")
    if not args.keep_edge:
        n0 = len(manifest)
        manifest = manifest[manifest["edge_flag"] == 0].reset_index(drop=True)
        print(f"Excluded {n0 - len(manifest)} edge segments "
              f"({n0} -> {len(manifest)})")

    results = {"generated": datetime.now(timezone.utc).isoformat(),
               "n_segments": int(len(manifest)),
               "conditions": [c["name"] for c in conds],
               "note": ("noise is added to the complex signal before the "
                        "transform, so SNR results are signal-domain; Doppler "
                        "blur is a representation-domain operation"),
               "scores": {}, "stability": {}, "importance": {}}
    clean_df = None
    t0 = time.perf_counter()

    for cond in conds:
        t1 = time.perf_counter()
        df = build_condition(manifest, cond, args.n_jobs)
        if cond["name"] == "clean":
            clean_df = df
        else:
            results["stability"][cond["name"]] = feature_stability(clean_df, df)
        results["scores"][cond["name"]] = {}
        for key in args.models:
            m = evaluate(df, key)
            if m:
                results["scores"][cond["name"]][key] = m
        if not args.skip_importance and cond["name"] in ("clean", "snr+0dB"):
            results["importance"][cond["name"]] = {
                key: permutation_importance(df, key) for key in args.models}
        print(f"  {cond['name']:<12} "
              + "  ".join(f"{k} {results['scores'][cond['name']][k]['mean_macro_f1']:.4f}"
                          for k in args.models
                          if k in results["scores"][cond["name"]])
              + f"   ({time.perf_counter() - t1:.0f}s)")

    print(f"\n=== macro-F1 by condition (degradation from clean) ===")
    base = {k: results["scores"]["clean"][k]["mean_macro_f1"]
            for k in args.models if k in results["scores"]["clean"]}
    hdr = f"{'condition':<12}" + "".join(f"{k:>18}" for k in base)
    print(hdr)
    for cond in conds:
        row = f"{cond['name']:<12}"
        for k in base:
            s = results["scores"][cond["name"]].get(k)
            row += (f"{s['mean_macro_f1']:>10.4f}"
                    f"{s['mean_macro_f1'] - base[k]:>+8.4f}") if s else " " * 18
        print(row)

    if results["stability"]:
        print("\n=== feature stability at 0 dB (Spearman clean vs corrupted) ===")
        st = results["stability"].get("snr+0dB", {})
        for name, v in sorted(st.items(), key=lambda kv: kv[1]["spearman"]):
            print(f"  {name:<26} rho {v['spearman']:>7.3f}   "
                  f"median shift {v['median_abs_shift']:>7.3f}")

    if results["importance"]:
        print("\n=== permutation importance, clean vs 0 dB ===")
        for key in args.models:
            c = results["importance"].get("clean", {}).get(key, {})
            n = results["importance"].get("snr+0dB", {}).get(key, {})
            if not c:
                continue
            print(f"  {key}")
            for name in sorted(c, key=lambda x: -c[x]):
                print(f"    {name:<26}{c[name]:>8.4f}{n.get(name, float('nan')):>10.4f}"
                      f"{n.get(name, 0) - c[name]:>+9.4f}")

    out = ARTIFACT_DIR / "results_robustness.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min | wrote {out.name}")


if __name__ == "__main__":
    main()
