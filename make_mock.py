"""
make_mock.py -- builds a miniature dataset in the exact layout of the real
files so the corrected chain can be executed end to end without the 1.6 GB
download.

Each mock measurement is an object array of 6 entries matching the diagnostic
output: label string, complex (1280, n_segments), then range, time, author
split and edge flag as (n_segments, 1) columns.

Targets are physically plausible at the real radar parameters so that the
reflector gate in validate_features.py is a genuine test rather than a
formality:
    reflector   stationary, all power at DC
    human       slow bulk motion plus limb modulation
    bird        moderate bulk motion plus slow wingbeat
    drone       bulk motion plus fast rotor modulation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

C = 299_792_458.0
FC = 77e9
LAM = C / FC
PRF = 17e3
N_SLOW = 256
N_RANGE = 5

SPEC = {
    "D1": dict(v=-6.0, radius=0.12, rot=14.0, blades=2, amp=0.45, n=5),
    "D3": dict(v=-9.0, radius=0.10, rot=18.0, blades=4, amp=0.40, n=5),
    "human_walk": dict(v=-1.4, radius=0.35, rot=1.0, blades=2, amp=0.30, n=5),
    "human_run": dict(v=-3.0, radius=0.40, rot=1.6, blades=2, amp=0.35, n=5),
    "seagull": dict(v=-7.0, radius=0.30, rot=3.5, blades=2, amp=0.35, n=5),
    "heron": dict(v=-5.0, radius=0.40, rot=2.5, blades=2, amp=0.35, n=5),
    "CR": dict(v=0.0, radius=0.0, rot=0.0, blades=0, amp=0.0, n=5),
}
# repeat each label so every class has >= 5 measurements for grouped folds
REPEATS = 3


def segment(spec: dict, t0: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N_SLOW) / PRF + t0
    k = -4.0 * np.pi / LAM
    r_body = 100.0 + spec["v"] * t
    x = np.exp(1j * k * r_body)
    for i in range(spec["blades"]):
        phi = 2.0 * np.pi * i / max(spec["blades"], 1)
        r = r_body + spec["radius"] * np.cos(2 * np.pi * spec["rot"] * t + phi)
        x = x + spec["amp"] * np.exp(1j * k * r)
    # five range cells: target centred on cell 2, weaker spill either side
    weights = np.array([0.15, 0.55, 1.0, 0.55, 0.15])
    out = weights[:, None] * x[None, :]
    out = out + 0.01 * (rng.standard_normal(out.shape)
                        + 1j * rng.standard_normal(out.shape))
    return out


def build(out_dir: Path, seed: int = 2026) -> None:
    # Refuse to write into a directory that already holds measurement files.
    # The mock uses the same filenames as the real data, so pointing this at
    # the real DATA_DIR would overwrite genuine measurements.
    if out_dir.exists():
        existing = list(out_dir.glob("raw_iq_measurement_*.npy"))
        if existing:
            raise SystemExit(
                f"REFUSING to write mock data into {out_dir}: it already "
                f"contains {len(existing)} measurement file(s).\n"
                f"The mock uses identical filenames and would overwrite real "
                f"data. Pass an empty or new directory, e.g. data/zenodo_mock."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = 0
    for label, spec in SPEC.items():
        for rep in range(REPEATS):
            n_seg = int(rng.integers(30, 60))
            cols = []
            for s in range(n_seg):
                seg = segment(spec, t0=s * 0.1, rng=rng)
                cols.append(seg.reshape(-1))          # (1280,) range-major
            iq = np.stack(cols, axis=1)               # (1280, n_seg)

            arr = np.empty(6, dtype=object)
            arr[0] = np.array([label], dtype="<U32")
            arr[1] = iq
            arr[2] = np.full((n_seg, 1), 100.0 + idx, dtype=np.float64)
            arr[3] = (np.arange(n_seg) * 0.1).reshape(-1, 1).astype(np.float64)
            arr[4] = np.full((n_seg, 1), 1 + (rep % 3), dtype=np.uint8)
            arr[5] = (rng.random((n_seg, 1)) < 0.002).astype(np.uint8)

            np.save(out_dir / f"raw_iq_measurement_{idx:03d}.npy", arr,
                    allow_pickle=True)
            idx += 1
    print(f"wrote {idx} mock measurements to {out_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python make_mock.py <empty output dir>\n"
            "  e.g. python make_mock.py data/zenodo_mock\n"
            "  Never point this at the directory holding your real data."
        )
    build(Path(sys.argv[1]))
