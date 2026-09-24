"""
src/s1_physics/zero_doppler_sweep.py

Decides ZERO_DOPPLER_HALFWIDTH from data rather than from a reading of the
protocol sentence.

The protocol says the zero-Doppler region is "the central 15% of the
normalised Doppler axis", which reads two ways:

    abs(f) <= 0.150   77 of 256 bins   everything under +/-2.48 m/s
    abs(f) <= 0.075   38 of 256 bins   everything under +/-1.24 m/s

The first run used 0.150 and gave medians reflector 1.000, human 0.961,
drone 0.926, bird 0.204. Drone, human and reflector are almost on top of each
other, so the descriptor is separating birds rather than static from moving.
This sweeps the half-width and reports what each choice buys.

Only zero_doppler_ratio depends on this constant, so nothing else needs
recomputing and the sweep runs off the existing memmap in under a minute.

Separability is reported as rank-based pairwise AUC, the probability that a
random sample of class A scores above a random sample of class B. 0.5 is no
separation, 1.0 or 0.0 is perfect. It is computed from the Mann-Whitney U
statistic, so it is robust to the heavy tails in this data and needs no
distributional assumption.

Run from the project root:
    python src/s1_physics/zero_doppler_sweep.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from config import (ARTIFACT_DIR, MEMMAP_DTYPE, MEMMAP_NAME, N_DOPPLER,
                    N_RANGE, WORK_DIR, banner)

HALFWIDTHS = (0.050, 0.075, 0.100, 0.125, 0.150, 0.200)
CLASS_ORDER = ("drone", "bird", "human", "reflector")


def pairwise_auc(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """P(x[a] > x[b]) from ranks. 0.5 means the two classes are inseparable."""
    xa, xb = x[a], x[b]
    n_a, n_b = len(xa), len(xb)
    if n_a == 0 or n_b == 0:
        return float("nan")
    r = rankdata(np.concatenate([xa, xb]))
    u = r[:n_a].sum() - n_a * (n_a + 1) / 2.0
    return float(u / (n_a * n_b))


def main() -> None:
    print(banner(), "\n")
    manifest = pd.read_csv(ARTIFACT_DIR / "dataset_manifest.csv")
    n = len(manifest)
    mm = np.memmap(WORK_DIR / MEMMAP_NAME, dtype=MEMMAP_DTYPE, mode="r",
                   shape=(n, N_RANGE, N_DOPPLER))

    f = np.fft.fftshift(np.fft.fftfreq(N_DOPPLER, d=1.0)) * 2.0
    lam = 299_792_458.0 / 77e9
    prf = 17e3

    print("Computing Doppler profiles...")
    profiles = np.asarray(mm).sum(axis=1)                    # (n, 256)
    totals = profiles.sum(axis=1, keepdims=True)
    p = profiles / np.maximum(totals, 1e-20)
    del profiles

    cls = manifest["class_name"].to_numpy()
    idx = {c: np.flatnonzero(cls == c) for c in CLASS_ORDER}

    rows = []
    for hw in HALFWIDTHS:
        v_ms = hw * (prf / 2.0) * lam / 2.0
        zd = p[:, np.abs(f) <= hw].sum(axis=1)

        med = {c: float(np.median(zd[idx[c]])) for c in CLASS_ORDER}
        aucs = {}
        for a, b in combinations(CLASS_ORDER, 2):
            auc = pairwise_auc(zd, idx[a], idx[b])
            aucs[f"{a[:4]}/{b[:4]}"] = auc
        # separability: mean distance from 0.5, doubled so 1.0 is perfect
        sep = float(np.mean([abs(v - 0.5) for v in aucs.values()]) * 2)

        rows.append(
            {
                "halfwidth": hw,
                "bins": int((np.abs(f) <= hw).sum()),
                "v_ms": round(v_ms, 2),
                **{f"med_{c}": round(med[c], 3) for c in CLASS_ORDER},
                **{f"auc_{k}": round(v, 3) for k, v in aucs.items()},
                "separability": round(sep, 3),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 50)

    print("\nPer-class medians of zero_doppler_ratio")
    print(df[["halfwidth", "bins", "v_ms"]
             + [f"med_{c}" for c in CLASS_ORDER]].to_string(index=False))

    print("\nPairwise separability (AUC; 0.5 = inseparable)")
    print(df[["halfwidth"] + [c for c in df.columns if c.startswith("auc_")]
             + ["separability"]].to_string(index=False))

    best = df.loc[df["separability"].idxmax()]
    print(f"\nHighest overall separability: halfwidth {best['halfwidth']} "
          f"({int(best['bins'])} bins, +/-{best['v_ms']} m/s), "
          f"score {best['separability']}")

    for pair in ("auc_dron/refl", "auc_dron/huma"):
        if pair in df.columns:
            r = df.loc[(df[pair] - 0.5).abs().idxmax()]
            print(f"  best for {pair}: halfwidth {r['halfwidth']} "
                  f"(AUC {r[pair]})")

    out = ARTIFACT_DIR / "zero_doppler_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out.name}")
    print("\nThe protocol permits 0.150 and 0.075. Report both in the paper; "
          "pick one for the frozen protocol and send Dr Salman this table.")


if __name__ == "__main__":
    main()
