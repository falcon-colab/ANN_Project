"""
convert_dataset.py

Step zero of the pipeline: unpack the Zenodo master file into the 130
per-measurement files that everything downstream reads.

The Zenodo download is a single object array (data_SAAB_SIRS_77GHz_FMCW.npy,
about 1.5 GB) holding 130 measurements. Each measurement is itself a 6-entry
object array: label string, complex I/Q of shape (1280, n_segments), range,
segment time, the authors' train/val/test indicator, and a field-of-view-edge
flag. This script writes each one to DATA_DIR/raw_iq_measurement_NNN.npy.

Corrections against the first version, which was written before the paths were
settled:
  1. The source path was hardcoded to C:\\Users\\Mux\\Documents\\ANN_Project,
     so the script broke the moment the project folder moved. It now resolves
     DATA_DIR from config.py like every other script, and takes --source for
     the master file.
  2. The output directory is created if missing, rather than failing on the
     first np.save.
  3. It refuses to overwrite an existing set unless --force is given. The 130
     files are the input to a frozen manifest, so silently rewriting them is
     exactly the kind of thing that invalidates a fold hash without anyone
     noticing.
  4. It reports the label of each measurement as it writes, which is the cheap
     sanity check that the object array was unpacked along the right axis.

Usage, from the project root:

    python convert_dataset.py --source path\\to\\data_SAAB_SIRS_77GHz_FMCW.npy

Zenodo record: 10.5281/zenodo.5845259 (Karlsson et al., CC-BY-4.0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "src" / "common"))

from config import DATA_DIR, banner  # noqa: E402

N_EXPECTED = 130
DEFAULT_SOURCE_NAME = "data_SAAB_SIRS_77GHz_FMCW.npy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=None,
                    help=f"the Zenodo master file; searched for by name "
                         f"({DEFAULT_SOURCE_NAME}) near the project root if "
                         f"not given")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing set of per-measurement files")
    args = ap.parse_args()

    print(banner(), "\n")

    source = args.source
    if source is None:
        here = Path(__file__).resolve().parent
        for candidate in (here / DEFAULT_SOURCE_NAME,
                          here / "data" / DEFAULT_SOURCE_NAME,
                          here / "artifacts" / DEFAULT_SOURCE_NAME,
                          DATA_DIR / DEFAULT_SOURCE_NAME,
                          DATA_DIR.parent / DEFAULT_SOURCE_NAME):
            if candidate.exists():
                source = candidate
                break
    if source is None or not source.exists():
        print(f"ERROR: master file not found. Pass --source explicitly.\n"
              f"       Expected something like {DEFAULT_SOURCE_NAME}")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(DATA_DIR.glob("raw_iq_measurement_*.npy"))
    if existing and not args.force:
        print(f"{len(existing)} per-measurement files already exist in "
              f"{DATA_DIR}.\nNothing written. Re-run with --force only if you "
              f"mean to replace them; the fold manifest was frozen against "
              f"the current set.")
        return 0

    print(f"source : {source}  ({source.stat().st_size / 1e9:.2f} GB)")
    print(f"target : {DATA_DIR}\nloading (this needs a few GB of RAM)...",
          flush=True)
    master = np.load(source, allow_pickle=True)
    print(f"loaded {len(master)} measurements\n")

    if len(master) != N_EXPECTED:
        print(f"WARNING: expected {N_EXPECTED} measurements, found "
              f"{len(master)}. Check that the download is complete before "
              f"treating anything downstream as valid.\n")

    for idx, measurement in enumerate(master):
        out = DATA_DIR / f"raw_iq_measurement_{idx:03d}.npy"
        np.save(out, measurement, allow_pickle=True)
        try:
            label = str(np.asarray(measurement[0]).reshape(-1)[0]).strip()
            shape = np.asarray(measurement[1]).shape
        except Exception:                      # noqa: BLE001
            label, shape = "unreadable", ()
        print(f"  {out.name}: label {label!r}, I/Q {shape}")

    n = len(sorted(DATA_DIR.glob("raw_iq_measurement_*.npy")))
    print(f"\nWrote {n} files to {DATA_DIR}")
    print("Next: python src/common/generate_manifest.py  (ONCE, write-once),")
    print("      then preprocess_iq.py and feature_extractor.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
