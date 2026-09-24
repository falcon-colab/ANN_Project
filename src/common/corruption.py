"""
src/common/corruption.py

Robustness perturbations, protocol section 17.

Two changes worth noting.

Seeding: the seed is derived from the segment_uid, not carried in a generator
that is threaded through a loop. The protocol requires that all four tracks
evaluate on identical corrupted samples. A shared generator only gives that if
every track iterates in exactly the same order, which nothing enforces. A uid
derived seed makes the corrupted version of a given segment identical
regardless of who computes it, in what order, or how many times.

Domain: noise is applied to the complex slow-time signal before the DFT. The
original protocol's image-domain wording ("representation-domain perturbation",
"must not be called physical radar noise") existed because DIAT-uSAT supplied
rendered images. With complex samples this is genuine signal-domain noise at a
defined SNR and may be described as such. Doppler blur remains a
representation-domain operation and keeps the original hedged wording.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter1d

CORRUPTION_SEED = 2026

SNR_LEVELS_DB = (-10.0, -5.0, 0.0, 5.0, 10.0)
BLUR_SIGMAS_PX = (1.0, 2.0, 4.0)
COMBINED = {"snr_db": 0.0, "sigma_px": 2.0}


def segment_rng(segment_uid: str, tag: str = "") -> np.random.Generator:
    """
    Deterministic generator for one segment and one condition.

    blake2b rather than md5: md5 is disabled on FIPS-enabled builds, which
    would make this crash on some cluster images and not on your laptop.
    """
    key = f"{CORRUPTION_SEED}|{segment_uid}|{tag}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def add_signal_noise(iq: np.ndarray, snr_db: float, segment_uid: str) -> np.ndarray:
    """
    Circularly symmetric complex Gaussian noise at a target SNR.

    Variance is split evenly between the real and imaginary parts so that the
    total complex noise power equals signal_power / 10**(snr_db/10).
    """
    iq = np.asarray(iq)
    signal_power = float(np.mean(np.abs(iq) ** 2))
    if signal_power <= 0.0:
        return iq.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(noise_power / 2.0)
    rng = segment_rng(segment_uid, f"noise{snr_db:g}")
    noise = scale * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape)
    )
    return iq + noise


def doppler_blur(power: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Gaussian blur along the Doppler axis only.

    `power` is (n_range, n_doppler), so Doppler is the last axis. Blurring the
    range axis would mix independent range cells and is not what section 17
    specifies. Deterministic, so it takes no seed.
    """
    return gaussian_filter1d(np.asarray(power, dtype=float), sigma=sigma_px, axis=-1)


def corrupt(iq: np.ndarray, segment_uid: str, snr_db=None, sigma_px=None,
            power_fn=None) -> np.ndarray:
    """
    Full chain: optional signal-domain noise, transform, optional Doppler blur.

    power_fn converts complex (n_range, n_doppler) to linear normalised power.
    Pass preprocess_iq.linear_power so the clean and corrupted paths use one
    implementation and cannot drift apart.
    """
    if power_fn is None:
        from preprocess_iq import linear_power as power_fn
    x = iq if snr_db is None else add_signal_noise(iq, snr_db, segment_uid)
    p = power_fn(x)
    return p if sigma_px is None else doppler_blur(p, sigma_px)


def conditions() -> list[dict]:
    """Every condition section 17 requires, as a list of kwargs for corrupt()."""
    out = [{"name": "clean", "snr_db": None, "sigma_px": None}]
    out += [{"name": f"snr{s:+g}dB", "snr_db": s, "sigma_px": None}
            for s in SNR_LEVELS_DB]
    out += [{"name": f"blur{g:g}px", "snr_db": None, "sigma_px": g}
            for g in BLUR_SIGMAS_PX]
    out += [{"name": "combined", "snr_db": COMBINED["snr_db"],
             "sigma_px": COMBINED["sigma_px"]}]
    return out
