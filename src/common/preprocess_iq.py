"""
src/common/preprocess_iq.py

Converts the complex slow-time segments into linear, scale-invariant Doppler
power and stores them as one memory-mapped float32 array whose row order
matches dataset_manifest.csv exactly.

    P = |FFT(x)|^2 / mean(|FFT(x)|^2)

No log. No min-max. No pseudo-power. The data is complex, so |FFT|^2 is
genuine power; the mean division removes the uncalibrated absolute level and
the range dependence, which is the only normalisation the descriptors need.

Bulk-Doppler centring is deliberately NOT applied here. Centring forces every
target's dominant return to DC, which would make the reflector validation gate
pass for a broken pipeline as easily as a correct one. Validate uncentred
first; centring is a separate, later change with its own before-and-after run.

Vectorised over segments: one FFT call per measurement rather than per
segment, which turns a ~76,000-iteration Python loop into 130 array ops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (ARTIFACT_DIR, DATA_DIR, MEMMAP_DTYPE, MEMMAP_NAME,
                    N_DOPPLER, N_RANGE, WORK_DIR, banner)

EPS = 1e-20


def linear_power(iq: np.ndarray) -> np.ndarray:
    """
    iq : complex, shape (..., N_RANGE, N_DOPPLER), slow time on the last axis.
    returns: float64 power, same shape, each segment normalised by its own mean.

    fftshift puts zero Doppler at index N_DOPPLER // 2 = 128.
    """
    spec = np.fft.fftshift(np.fft.fft(iq, axis=-1), axes=-1)
    power = np.abs(spec) ** 2
    mean = power.mean(axis=(-2, -1), keepdims=True)
    return power / (mean + EPS)


def unpack_measurement(path: Path) -> np.ndarray:
    """
    Return complex segments shaped (n_segments, N_RANGE, N_DOPPLER).

    The stored column is (1280, n_segments) with the first 256 entries being
    the first range cell, so the reshape is C-order over (range, slow time).
    """
    raw = np.load(path, allow_pickle=True)
    flat = np.asarray(raw[1]).T                      # (n_segments, 1280)
    if flat.shape[1] != N_RANGE * N_DOPPLER:
        raise ValueError(
            f"{path.name}: expected {N_RANGE * N_DOPPLER} elements per segment, "
            f"got {flat.shape[1]}"
        )
    return flat.reshape(-1, N_RANGE, N_DOPPLER)


def main() -> None:
    print(banner(), "\n")
    manifest = pd.read_csv(ARTIFACT_DIR / "dataset_manifest.csv")
    meta = json.loads((ARTIFACT_DIR / "manifest_meta.json").read_text())

    n_rows = len(manifest)
    if meta["n_rows"] != n_rows:
        raise ValueError(
            "manifest_meta.json disagrees with dataset_manifest.csv; "
            "re-run generate_manifest.py"
        )

    # WORK_DIR, never DATA_DIR: on Atlas the data directory is shared and may
    # be read-only, and a 389 MB per-student artefact does not belong there.
    mm_path = WORK_DIR / MEMMAP_NAME
    mm = np.memmap(mm_path, dtype=MEMMAP_DTYPE, mode="w+",
                   shape=(n_rows, N_RANGE, N_DOPPLER))

    # Row order in the manifest is the memmap row order. Positions are taken
    # from the manifest itself rather than assumed, so the mapping holds even
    # if the manifest is ever re-sorted.
    manifest = manifest.reset_index().rename(columns={"index": "row"})

    done = 0
    for measurement_id, group in manifest.groupby("measurement_id", sort=False):
        segments = unpack_measurement(DATA_DIR / measurement_id)
        seg_idx = group["segment_index"].to_numpy()
        rows = group["row"].to_numpy()
        if seg_idx.max() >= segments.shape[0]:
            raise ValueError(f"{measurement_id}: manifest indexes past the data")
        mm[rows] = linear_power(segments[seg_idx]).astype("float32")
        done += len(rows)
        if done % 10_000 < len(rows):
            print(f"  {done}/{n_rows} segments")

    mm.flush()
    del mm

    check = np.memmap(mm_path, dtype=MEMMAP_DTYPE, mode="r",
                      shape=(n_rows, N_RANGE, N_DOPPLER))
    sample = np.asarray(check[: min(512, n_rows)])
    print(f"\nWrote {mm_path.name}  shape=({n_rows}, {N_RANGE}, {N_DOPPLER})")
    print(f"  min {sample.min():.4g}  max {sample.max():.4g}  "
          f"mean {sample.mean():.4g}")
    assert np.all(sample >= 0), "negative power written"
    assert np.all(np.isfinite(sample)), "non-finite power written"
    print("  sanity: nonnegative and finite")


if __name__ == "__main__":
    main()
