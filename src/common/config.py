"""
src/common/config.py

One place where every path, device and worker count is decided, so the same
code runs unchanged on a local Windows PC and on the Atlas cluster.

Three directories, not one, following the Atlas wiki:

  DATA_DIR      the 130 raw measurement files. On Atlas the job stages these
                onto the compute node's /scratch, so reads come off a local
                SSD instead of over the network.
  WORK_DIR      large, regenerable, node-local. The ~389 MB memmap lives here,
                also on /scratch. It is rebuilt every job, so losing it when
                the node changes costs nothing.
  ARTIFACT_DIR  small and persistent: manifests, the features CSV, models,
                results, plots. On Atlas this belongs on /data, the networked
                backed-up filesystem, NOT on /home, which the wiki reserves
                for scripts and logs and which is not synced between the two
                login nodes.

Select with PROJECT_ENV:

    local (default)   everything under the repo
    atlas             defaults below, each overridable by ATLAS_DATA_DIR,
                      ATLAS_WORK_DIR and ATLAS_ARTIFACT_DIR

    Windows PowerShell:  $env:PROJECT_ENV = "atlas"
    Linux / SLURM:       export PROJECT_ENV=atlas
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENVIRONMENT = os.getenv("PROJECT_ENV", "local").strip().lower()

if ENVIRONMENT == "atlas":
    _user = os.getenv("USER", "student")
    DATA_DIR = Path(os.getenv("ATLAS_DATA_DIR",
                              f"/scratch/{_user}/se801/zenodo"))
    WORK_DIR = Path(os.getenv("ATLAS_WORK_DIR",
                              f"/scratch/{_user}/se801/work"))
    ARTIFACT_DIR = Path(os.getenv("ATLAS_ARTIFACT_DIR",
                                  f"/data/{_user}/se801/artifacts"))
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    PIN_MEMORY = True
else:
    DATA_DIR = BASE_DIR / "data" / "zenodo"
    WORK_DIR = BASE_DIR / "data" / "work"
    ARTIFACT_DIR = BASE_DIR / "data"
    BATCH_SIZE = 16
    NUM_WORKERS = 0          # 0 on Windows: avoids DataLoader pipe deadlocks
    PIN_MEMORY = False

# Creating these is best effort. On Atlas, /scratch exists only inside a batch
# job, so importing this module on a login node cannot create it and must not
# crash; the job script creates both before any Python runs.
for _d in (WORK_DIR, ARTIFACT_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError as _e:
        print(f"note: cannot create {_d} ({type(_e).__name__}); "
              f"expected on a login node", flush=True)

# Protocol constants
SEED = 2026
OUTER_FOLD_SEEDS = (2026, 2027, 2028, 2029, 2030)   # training seeds, per fold
N_OUTER = 5
N_INNER = 5
N_CLASSES = 4

# Explicit little-endian float32. Native '=f4' would silently produce a file
# another machine reads as garbage; this makes the memmap portable.
MEMMAP_DTYPE = "<f4"
MEMMAP_NAME = "processed_linear_power.dat"

N_RANGE = 5
N_DOPPLER = 256


def device():
    """Resolved at call time so importing this module never requires torch."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def banner() -> str:
    return (
        f"PROJECT_ENV={ENVIRONMENT}\n"
        f"  DATA_DIR     {DATA_DIR}\n"
        f"  WORK_DIR     {WORK_DIR}\n"
        f"  ARTIFACT_DIR {ARTIFACT_DIR}"
    )


if __name__ == "__main__":
    print(banner())
    print(f"  DATA_DIR exists: {DATA_DIR.exists()}")
