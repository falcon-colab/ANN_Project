"""
src/s1_physics/feature_extractor.py

The ten protocol-mandated physics-guided descriptors, computed per 15 ms
segment.

Corrections against the previous draft, each of which changed feature values
silently rather than raising an error:

  1. skewness / kurtosis are now POWER-WEIGHTED central moments over the
     Doppler axis. scipy.stats.skew(profile) treats the 256 power values as
     independent observations and is blind to the sign of Doppler asymmetry:
     a right-tailed and a left-tailed spectrum return the identical number.
  2. doppler_bandwidth_80 is now the width of the central 80% interval of the
     Doppler CDF. Growing a symmetric window around bin 128 measured distance
     from zero Doppler, i.e. radial velocity, not bandwidth. Two spectra of
     identical width returned 18 and 166 bins depending only on target speed.
  3. side_lobe_energy_ratio is anchored on the DOMINANT Doppler peak, not on
     zero Doppler. Defined against the zero-Doppler band it was exactly
     1 - zero_doppler_ratio, correlation -1.000, so two of the ten descriptors
     carried one number between them.
  4. The temporal descriptors use an explicit in-segment STFT.
     scipy.signal.spectrogram defaults to detrend='constant', which subtracts
     the per-window mean and therefore annihilates the DC component. On a
     stationary target that zeroes the signal entirely. Its output is also
     already power under mode='psd', so |Sxx|**2 squared power a second time.
  5. temporal_energy_variance is the squared coefficient of variation, so it
     measures fluctuation depth rather than absolute return level.
  6. Entropies are normalised by log(n_bins) to land in [0, 1], so the
     Doppler-axis and side-lobe entropies remain comparable to each other and
     across any later change of transform length.
  7. The Doppler axis is normalised to [-1, 1), matching the protocol, so the
     moment features do not change scale if the transform length changes.

The range axis is collapsed by summing power over the five cells: total power
at each Doppler across the target's range extent. A per-bin max would stitch
together different range cells along the Doppler axis and is not a spectrum of
anything physical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from config import (ARTIFACT_DIR, DATA_DIR, MEMMAP_DTYPE, MEMMAP_NAME,
                    N_DOPPLER, N_RANGE, WORK_DIR, banner)

EPS = 1e-20

# --- Frozen feature parameters. These are protocol choices, not details. ----
# Half-width of the zero-Doppler band on the normalised [-1, 1) Doppler axis.
#
# The protocol says "the central 15% of the normalised Doppler axis", which
# admits two readings, and they are not close together in physical terms:
#
#   0.150  -> 15% each side, 77 of 256 bins, everything under +/-2.48 m/s
#   0.075  -> 15% of the full axis, 38 bins, everything under +/-1.24 m/s
#
# A walking human has a radial speed near 1.4 m/s, so the wider reading files
# walking humans as zero-Doppler targets and the narrower one does not. This
# must be confirmed with Dr Salman before the freeze rather than assumed.
#ZERO_DOPPLER_HALFWIDTH = 0.15
ZERO_DOPPLER_HALFWIDTH = 0.075

# Half-width of the main lobe about the dominant Doppler peak, same units.
# Not specified by the protocol; needs sign-off.
MAIN_LOBE_HALFWIDTH = 0.10

BANDWIDTH_QUANTILES = (0.10, 0.90)
PEAK_SMOOTH_BINS = 3

STFT_NPERSEG = 64
STFT_NOVERLAP = 48          # 75% overlap -> 13 frames from 256 slow-time samples

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


def doppler_axis(n: int = N_DOPPLER) -> np.ndarray:
    """Normalised Doppler in [-1, 1); +/-1 is +/- PRF/2. DC sits at index n//2."""
    return np.fft.fftshift(np.fft.fftfreq(n, d=1.0)) * 2.0


F_AXIS = doppler_axis()


def _pmf(v: np.ndarray) -> np.ndarray:
    total = float(v.sum())
    if total <= EPS:
        return np.full_like(v, np.nan, dtype=float)
    return v / total


def _norm_entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats divided by log(len(p)), so the range is [0, 1]."""
    if p.size < 2 or not np.all(np.isfinite(p)):
        return float("nan")
    q = p[p > EPS]
    if q.size == 0:
        return float("nan")
    return float(-np.sum(q * np.log(q)) / np.log(p.size))


def _quantile(p: np.ndarray, f: np.ndarray, q: float) -> float:
    c = np.cumsum(p)
    if c[-1] <= 0:
        return float("nan")
    return float(np.interp(q, c / c[-1], f))


def _smooth(v: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or v.size < width:
        return v
    return np.convolve(v, np.ones(width) / width, mode="same")


def stft_power(iq: np.ndarray) -> np.ndarray:
    """
    Explicit in-segment STFT. iq is complex (N_RANGE, N_DOPPLER).

    Returns power of shape (n_freq, n_frames), summed over range cells.
    Written out rather than calling scipy so that no detrending happens and
    so the operation is defensible line by line in the code demonstration.
    """
    step = STFT_NPERSEG - STFT_NOVERLAP
    n_frames = 1 + (iq.shape[-1] - STFT_NPERSEG) // step
    window = np.hanning(STFT_NPERSEG)
    frames = np.stack(
        [iq[:, i * step : i * step + STFT_NPERSEG] * window for i in range(n_frames)],
        axis=-1,
    )                                              # (N_RANGE, NPERSEG, n_frames)
    spec = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    return (np.abs(spec) ** 2).sum(axis=0)         # (NPERSEG, n_frames)


def extract_features(power_5x256: np.ndarray, iq_5x256: np.ndarray) -> dict:
    """
    power_5x256 : linear normalised power from preprocess_iq.linear_power
    iq_5x256    : the matching complex segment, used only for the STFT
    """
    out: dict[str, float] = {}

    # ---- Doppler-axis descriptors --------------------------------------
    profile = np.asarray(power_5x256, dtype=float).sum(axis=0)   # sum over range
    p = _pmf(profile)
    f = F_AXIS

    if np.all(np.isfinite(p)):
        mu = float(np.sum(p * f))
        var = float(np.sum(p * (f - mu) ** 2))
        sd = float(np.sqrt(max(var, 0.0)))
        out["doppler_spread"] = sd
        if sd > EPS:
            z = (f - mu) / sd
            out["skewness"] = float(np.sum(p * z**3))
            out["kurtosis"] = float(np.sum(p * z**4) - 3.0)   # excess kurtosis
        else:
            out["skewness"] = float("nan")
            out["kurtosis"] = float("nan")

        lo = _quantile(p, f, BANDWIDTH_QUANTILES[0])
        hi = _quantile(p, f, BANDWIDTH_QUANTILES[1])
        out["doppler_bandwidth_80"] = hi - lo

        out["spectral_entropy"] = _norm_entropy(p)

        out["zero_doppler_ratio"] = float(
            p[np.abs(f) <= ZERO_DOPPLER_HALFWIDTH].sum()
        )

        peak_f = f[int(np.argmax(_smooth(p, PEAK_SMOOTH_BINS)))]
        main = np.abs(f - peak_f) <= MAIN_LOBE_HALFWIDTH
        side_mass = float(p[~main].sum())
        out["side_lobe_energy_ratio"] = side_mass
        if side_mass > EPS and np.count_nonzero(~main) >= 2:
            out["side_lobe_entropy"] = _norm_entropy(p[~main] / side_mass)
        else:
            out["side_lobe_entropy"] = float("nan")
    else:
        for k in ("doppler_spread", "skewness", "kurtosis",
                  "doppler_bandwidth_80", "spectral_entropy",
                  "zero_doppler_ratio", "side_lobe_energy_ratio",
                  "side_lobe_entropy"):
            out[k] = float("nan")

    # ---- Temporal descriptors, from the in-segment STFT ------------------
    S = stft_power(np.asarray(iq_5x256))
    energy = S.sum(axis=0)                      # power per time frame
    mean_e = float(energy.mean())
    if energy.size < 2 or mean_e <= EPS:
        out["temporal_energy_variance"] = float("nan")
        out["temporal_entropy"] = float("nan")
    else:
        # squared coefficient of variation: scale invariant by construction
        out["temporal_energy_variance"] = float(np.var(energy / mean_e))
        out["temporal_entropy"] = _norm_entropy(_pmf(energy))

    return {k: out[k] for k in FEATURE_NAMES}


def main() -> None:
    print(banner(), "\n")
    manifest = pd.read_csv(ARTIFACT_DIR / "dataset_manifest.csv")
    meta = json.loads((ARTIFACT_DIR / "manifest_meta.json").read_text())
    n_rows = len(manifest)
    if meta["n_rows"] != n_rows:
        raise ValueError("manifest and sidecar disagree; re-run generate_manifest.py")

    mm = np.memmap(
        WORK_DIR / MEMMAP_NAME,
        dtype=MEMMAP_DTYPE, mode="r", shape=(n_rows, N_RANGE, N_DOPPLER),
    )
    manifest = manifest.reset_index().rename(columns={"index": "row"})

    rows_out = []
    for measurement_id, group in manifest.groupby("measurement_id", sort=False):
        raw = np.load(DATA_DIR / measurement_id, allow_pickle=True)
        iq_all = np.asarray(raw[1]).T.reshape(-1, N_RANGE, N_DOPPLER)
        for rec in group.itertuples(index=False):
            feats = extract_features(mm[rec.row], iq_all[rec.segment_index])
            feats["segment_uid"] = rec.segment_uid
            feats["measurement_id"] = rec.measurement_id
            feats["class_id"] = int(rec.class_id)
            rows_out.append(feats)
        print(f"  {measurement_id}: {len(group)} segments")

    df = pd.DataFrame(rows_out)
    out_path = ARTIFACT_DIR / "s1_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path.name}: {len(df)} rows, "
          f"{len(FEATURE_NAMES)} features")
    n_nan = int(df[list(FEATURE_NAMES)].isna().sum().sum())
    print(f"  NaN cells: {n_nan}")


if __name__ == "__main__":
    main()
