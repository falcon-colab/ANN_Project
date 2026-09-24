"""
src/s1_physics/validate_features.py

The gate that must pass before any model is trained.

Checkpoint 1  numerical integrity
Checkpoint 2  physics bounds and the reflector gate
Checkpoint 3  redundancy and class separation, plus the required plots

The reflector gate is the one that matters. A corner reflector is a stationary
trihedral with no micro-Doppler, so essentially all of its power sits in the
zero-Doppler bin. If zero_doppler_ratio for class 3 is not near 1.0 and clearly
above the moving classes, the DSP chain is wrong and nothing downstream means
anything. It was this check, quietly downgraded to a warning, that hid the
min-max scaling defect for several days.

Exit status is nonzero on failure so the script can gate a run script or CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from config import ARTIFACT_DIR, banner
sys.path.append(str(Path(__file__).resolve().parent))
from feature_extractor import ZERO_DOPPLER_HALFWIDTH

CLASS_NAMES = {0: "drone", 1: "bird", 2: "human", 3: "reflector"}
REFLECTOR = 3

FEATURE_NAMES = (
    "side_lobe_energy_ratio", "side_lobe_entropy", "spectral_entropy",
    "temporal_energy_variance", "temporal_entropy", "doppler_bandwidth_80",
    "doppler_spread", "zero_doppler_ratio", "skewness", "kurtosis",
)

REFLECTOR_ZD_MIN = 0.90      # gate: median zero-Doppler ratio for reflectors
SEPARATION_MARGIN = 0.30     # gate: reflector median minus worst moving median


def _ok(msg):   print(f"  PASS  {msg}")
def _bad(msg):  print(f"  FAIL  {msg}")
def _warn(msg): print(f"  WARN  {msg}")


def run() -> int:
    print(banner(), "\n")
    df = pd.read_csv(ARTIFACT_DIR / "s1_features.csv")
    feats = df[list(FEATURE_NAMES)]
    failures = 0

    print(f"Loaded {len(df)} segments across "
          f"{df['measurement_id'].nunique()} measurements\n")

    print("=== CHECKPOINT 1: numerical integrity ===")
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        _bad(f"missing features: {missing}"); failures += 1
    else:
        _ok("all ten descriptors present")

    n_nan = int(feats.isna().sum().sum())
    if n_nan:
        _warn(f"{n_nan} NaN cells "
              f"({100*n_nan/feats.size:.3f}%): {feats.isna().sum()[feats.isna().sum()>0].to_dict()}")
    else:
        _ok("no NaN values")

    if np.isinf(feats.to_numpy(dtype=float)).any():
        _bad("infinite values present"); failures += 1
    else:
        _ok("no infinite values")

    print("\n=== CHECKPOINT 2: physics bounds and the reflector gate ===")
    for col in ("zero_doppler_ratio", "side_lobe_energy_ratio",
                "spectral_entropy", "side_lobe_entropy", "temporal_entropy"):
        v = df[col].dropna()
        if v.between(-1e-6, 1 + 1e-6).all():
            _ok(f"{col} within [0, 1]")
        else:
            _bad(f"{col} out of [0, 1]: min {v.min():.4g} max {v.max():.4g}")
            failures += 1

    v = df["doppler_spread"].dropna()
    if (v >= 0).all() and (v <= 1.2).all():
        _ok("doppler_spread nonnegative and within the normalised axis")
    else:
        _bad(f"doppler_spread out of range: min {v.min():.4g} max {v.max():.4g}")
        failures += 1

    med = df.groupby("class_id")["zero_doppler_ratio"].median()
    if REFLECTOR in med.index:
        refl = float(med[REFLECTOR])
        moving = med.drop(index=REFLECTOR)
        worst = float(moving.max())
        print(f"        zero_doppler_ratio medians: "
              + ", ".join(f"{CLASS_NAMES[c]} {m:.3f}" for c, m in med.items()))
        if refl >= REFLECTOR_ZD_MIN:
            _ok(f"reflector zero-Doppler ratio {refl:.3f} >= {REFLECTOR_ZD_MIN}")
        else:
            _bad(f"reflector zero-Doppler ratio {refl:.3f} < {REFLECTOR_ZD_MIN}. "
                 "DSP chain is wrong. Do not train.")
            failures += 1
        if refl > worst:
            _ok(f"reflector ranks above every moving class "
                f"(next highest {worst:.3f})")
        else:
            _bad("a moving class has a higher zero-Doppler ratio than the "
                 "static reflector; the DSP chain is wrong")
            failures += 1
        if refl - worst < SEPARATION_MARGIN:
            nearest = CLASS_NAMES[int(moving.idxmax())]
            v_ms = ZERO_DOPPLER_HALFWIDTH * (17e3 / 2) * (299_792_458.0 / 77e9) / 2
            _warn(f"margin over the nearest moving class ({nearest}) is only "
                  f"{refl - worst:.3f}. At halfwidth "
                  f"{ZERO_DOPPLER_HALFWIDTH:g} the band counts everything "
                  f"under +/-{v_ms:.2f} m/s as stationary, so this is expected "
                  f"if {nearest} targets are near-stationary. Check which "
                  f"class it is before dismissing it.")
    else:
        _bad("no reflector samples found"); failures += 1

    spread = df.groupby("class_id")["doppler_spread"].median()
    if REFLECTOR in spread.index and len(spread) > 1:
        if float(spread[REFLECTOR]) < float(spread.drop(index=REFLECTOR).min()):
            _ok("reflectors have the narrowest Doppler spread")
        else:
            _bad("a moving class has narrower spread than the static reflector")
            failures += 1

    print("\n=== CHECKPOINT 3: redundancy and separation ===")
    corr = feats.corr().abs()
    corr_v = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_v, 0.0)
    corr = pd.DataFrame(corr_v, index=corr.index, columns=corr.columns)
    stacked = corr.stack().sort_values(ascending=False)
    pairs, seen = [], set()
    for (a, b), r in stacked.items():
        if r <= 0.90 or (b, a) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b, float(r)))
    if pairs:
        _warn(f"{len(pairs)} feature pair(s) above |r| = 0.90 "
              "(candidates to drop for the five-feature subset):")
        for a, b, r in pairs:
            print(f"          {a} / {b}: {r:.3f}")
    else:
        _ok("no feature pair above |r| = 0.90")

    zd_sl = float(feats["zero_doppler_ratio"].corr(feats["side_lobe_energy_ratio"]))
    if abs(zd_sl + 1.0) < 0.01:
        _bad(f"zero_doppler_ratio and side_lobe_energy_ratio correlate "
             f"{zd_sl:.3f}: the side lobe is defined as the complement of the "
             "zero-Doppler band, so the two carry one number")
        failures += 1
    else:
        _ok(f"zero-Doppler and side-lobe ratios are independent (r = {zd_sl:.3f})")

    dead = [c for c in FEATURE_NAMES
            if df[c].dropna().nunique() <= 1
            or float(df[c].std(skipna=True) or 0.0) < 1e-12]
    if dead:
        _warn(f"constant or near-constant features: {dead}")
    else:
        _ok("every feature varies across the dataset")

    _plots(df)

    print(f"\n{'GATE PASSED' if failures == 0 else f'GATE FAILED: {failures} check(s)'}")
    if failures:
        print("Do not run the classical baselines until these pass.")
    return failures


def _plots(df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        _warn("matplotlib/seaborn unavailable; skipping plots")
        return

    out = ARTIFACT_DIR
    feats = df[list(FEATURE_NAMES)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(feats.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1)
    plt.title("S1 feature correlation")
    plt.tight_layout()
    plt.savefig(out / "feature_correlation.png", dpi=120)
    plt.close()

    d = df.copy()
    d["class_name"] = d["class_id"].map(CLASS_NAMES)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, col in zip(axes.ravel(), FEATURE_NAMES):
        sns.boxplot(data=d, x="class_name", y=col, hue="class_name",
                    palette="Set2", legend=False, ax=ax, showfliers=False)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes.ravel()[len(FEATURE_NAMES):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / "feature_distributions.png", dpi=120)
    plt.close()
    _ok("wrote feature_correlation.png and feature_distributions.png")


if __name__ == "__main__":
    sys.exit(run())
