"""
src/s1_physics/feature_transforms.py

Two separate steps that are often confused. They are not alternatives and the
order matters.

Step 1, signed log compression. Fixed, parameterless, applied identically to
every row, fitted on nothing. It exists because two of the ten descriptors are
heavy tailed by construction: for a near-delta Doppler spectrum the standard
deviation is tiny, so the standardised third and fourth moments blow up. On
the real data kurtosis has a reflector median near 2,400 against roughly 0 for
the other classes, and skewness runs from -11 to +10.

    signed_log1p(x) = sign(x) * log1p(abs(x))

Signed rather than plain log1p because skewness is genuinely negative for
receding-tail spectra and the sign carries physical meaning. The transform is
monotone, so it preserves every ranking and therefore cannot help or harm the
tree models, which are rank based. It exists for RBF-SVM and the MLP, whose
distance metrics would otherwise be dominated by the reflector class on one
feature.

Step 2, standardisation. Fitted on the current training fold ONLY, then
applied to validation and test. This is the protocol's leakage rule, section 6.

Because step 1 is fitted on nothing, it can be applied before splitting
without leaking. Step 2 must be inside the fold loop. Keeping them in one
object makes it hard to get that wrong.

Whether the signed log helps is itself an ablation entry: run the classical
models with SIGNED_LOG=on and off and report both. The tree results should be
identical, which is a useful correctness check on the whole pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    "side_lobe_energy_ratio",
    "side_lobe_entropy",
    "spectral_entropy",
    "temporal_energy_variance",
    "temporal_entropy",
    "doppler_bandwidth_80",
    "doppler_spread",
    "zero_doppler_ratio",
    "skewness",
    "kurtosis",
)

# Only the standardised moments are heavy tailed. The ratios and entropies are
# already bounded in [0, 1] and the spreads are bounded by the Doppler axis.
HEAVY_TAILED = ("skewness", "kurtosis")

# The protocol asks for a compact five-feature subset. There are two defensible
# ways to choose one and they disagree, so both are defined here and both are
# reported; the report must not quietly present only the better of the two.
#
# 1. From the CORRELATION STRUCTURE, chosen before any model was trained: one
#    feature from the spread cluster (side_lobe_energy_ratio, spectral_entropy,
#    doppler_bandwidth_80 and doppler_spread correlate 0.85 to 0.94), one from
#    the temporal pair (0.97), plus the three that are largely independent.
#    This maximises coverage of the correlation graph and it lost 0.053 to
#    0.086 macro-F1 against the full ten on the development run.
#
# 2. From the PERMUTATION IMPORTANCE measured on the full-feature models, which
#    ranked spectral_entropy, doppler_spread and side_lobe_energy_ratio highest
#    for every model and skewness and temporal_energy_variance lowest. This
#    keeps the three strongest plus kurtosis and zero_doppler_ratio, and so
#    deliberately takes three members of one correlated cluster.
#
# Subset 2 is selected using the outer-fold models of the full-feature run, so
# it is not a clean held-out choice and is reported as such: it answers "how
# well can five features do" rather than "how well would five features
# generalise if chosen blind". Subset 1 answers the second question honestly.
# Keeping both, and saying which is which, is the only defensible reading.
COMPACT_SUBSET_CORRELATION = (
    "doppler_spread",
    "temporal_entropy",
    "zero_doppler_ratio",
    "side_lobe_entropy",
    "kurtosis",
)

COMPACT_SUBSET_IMPORTANCE = (
    "spectral_entropy",
    "doppler_spread",
    "side_lobe_energy_ratio",
    "kurtosis",
    "zero_doppler_ratio",
)

# What --features compact resolves to elsewhere in the codebase.
COMPACT_SUBSET = COMPACT_SUBSET_IMPORTANCE


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Monotone, sign preserving, fitted on nothing."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


class FeaturePreprocessor:
    """
    fit() must see only the training portion of the current fold.

    Usage inside a fold:
        pre = FeaturePreprocessor(signed_log=True)
        Xtr = pre.fit_transform(df_train)
        Xva = pre.transform(df_val)      # never fit on this
    """

    def __init__(self, features=FEATURE_NAMES, signed_log: bool = True):
        self.features = tuple(features)
        self.signed_log = signed_log
        self.scaler = StandardScaler()
        self._fitted = False

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")
        X = df[list(self.features)].to_numpy(dtype=float)
        if self.signed_log:
            cols = [i for i, c in enumerate(self.features) if c in HEAVY_TAILED]
            if cols:
                X = X.copy()
                X[:, cols] = signed_log1p(X[:, cols])
        return X

    def fit(self, df_train: pd.DataFrame) -> "FeaturePreprocessor":
        self.scaler.fit(self._raw(df_train))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() on the training fold before transform()")
        return self.scaler.transform(self._raw(df))

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        return self.fit(df_train).transform(df_train)


def describe_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Before and after ranges, to show why the compression is needed."""
    rows = []
    for c in FEATURE_NAMES:
        v = df[c].to_numpy(dtype=float)
        w = signed_log1p(v) if c in HEAVY_TAILED else v
        rows.append(
            {
                "feature": c,
                "compressed": c in HEAVY_TAILED,
                "raw_min": round(float(np.nanmin(v)), 3),
                "raw_max": round(float(np.nanmax(v)), 3),
                "raw_range": round(float(np.nanmax(v) - np.nanmin(v)), 3),
                "out_min": round(float(np.nanmin(w)), 3),
                "out_max": round(float(np.nanmax(w)), 3),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
    from config import ARTIFACT_DIR

    df = pd.read_csv(ARTIFACT_DIR / "s1_features.csv")
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(describe_effect(df).to_string(index=False))
