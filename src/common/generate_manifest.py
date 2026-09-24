"""
src/common/generate_manifest.py

Builds the segment-level dataset manifest and the nested, measurement-grouped,
stratified fold assignment, then verifies it before writing.

Changes from the previous version:
  - keeps `label_raw` (needed for the dataset card and for any
    leave-one-drone-type-out experiment; regenerating it later means
    re-reading 1.6 GB)
  - fails loudly on unmapped labels instead of silently dropping them
  - fold columns are pre-created so the frame is not mutated while the
    splitter is iterating over it
  - writes both dataset_manifest.csv and fold_manifest.csv, because the
    downstream scripts expect the latter by name
  - records the row count in a sidecar so the memmap can never be opened
    with the wrong shape
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append(str(Path(__file__).resolve().parent))
from config import (ARTIFACT_DIR, DATA_DIR, MEMMAP_DTYPE, N_CLASSES,
                    N_DOPPLER, N_INNER, N_OUTER, N_RANGE, SEED, banner)

EXPECTED_MEASUREMENTS = 130
EXPECTED_SEGMENTS = 75_868
EXPECTED_EDGE_FLAGGED = 67

CLASS_NAMES = {0: "drone", 1: "bird", 2: "human", 3: "reflector"}

BIRD_TOKENS = ("seagull", "gull", "pigeon", "raven", "heron")


def coarse_class(label_raw: str) -> int:
    """Map one of the 15 raw label strings onto the four coarse classes."""
    s = label_raw.strip().lower()
    if s.startswith("d") and s[1:].isdigit():
        return 0
    if any(tok in s for tok in BIRD_TOKENS):
        return 1
    if "human" in s:
        return 2
    if s == "cr":
        return 3
    raise ValueError(f"unmapped label: {label_raw!r}")


def _col(arr) -> np.ndarray:
    """Flatten an (n, 1) metadata column to (n,)."""
    return np.asarray(arr).ravel()


def build_records() -> pd.DataFrame:
    raw_files = sorted(DATA_DIR.glob("raw_iq_measurement_*.npy"))
    if not raw_files:
        raise FileNotFoundError(f"no raw measurement files under {DATA_DIR}")

    records = []
    for path in raw_files:
        raw = np.load(path, allow_pickle=True)
        label_raw = str(np.asarray(raw[0]).ravel()[0]).strip()
        class_id = coarse_class(label_raw)

        iq = raw[1]                       # (1280, n_segments)
        n_seg = iq.shape[1]
        rng_m = _col(raw[2])
        t_s = _col(raw[3])
        split = _col(raw[4])
        edge = _col(raw[5])

        for arr, name in ((rng_m, "range"), (t_s, "time"),
                          (split, "author_split"), (edge, "edge_flag")):
            if arr.size != n_seg:
                raise ValueError(
                    f"{path.name}: {name} has {arr.size} entries, "
                    f"expected {n_seg}"
                )

        for i in range(n_seg):
            records.append(
                {
                    "segment_uid": f"{path.stem}_seg{i:05d}",
                    "measurement_id": path.name,
                    "segment_index": i,
                    "label_raw": label_raw,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "range_m": float(rng_m[i]),
                    "time_s": float(t_s[i]),
                    "author_split": int(split[i]),
                    "edge_flag": int(edge[i]),
                }
            )

    return pd.DataFrame(records).reset_index(drop=True)


def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["outer_fold"] = -1
    for k in range(N_OUTER):
        df[f"inner_fold_for_outer_{k}"] = -1

    y = df["class_id"].to_numpy()
    groups = df["measurement_id"].to_numpy()
    X = np.zeros((len(df), 1))

    outer = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(X, y, groups))

    for k, (train_idx, test_idx) in enumerate(outer_splits):
        df.iloc[test_idx, df.columns.get_loc("outer_fold")] = k

        col = df.columns.get_loc(f"inner_fold_for_outer_{k}")
        inner = StratifiedGroupKFold(
            n_splits=N_INNER, shuffle=True, random_state=SEED + k
        )
        for j, (_, val_rel) in enumerate(
            inner.split(X[train_idx], y[train_idx], groups[train_idx])
        ):
            df.iloc[train_idx[val_rel], col] = j

    return df


def verify(df: pd.DataFrame) -> dict:
    assert (df["outer_fold"] >= 0).all(), "segments without an outer fold"

    spans = df.groupby("measurement_id")["outer_fold"].nunique()
    bad = spans[spans > 1].index.tolist()
    assert not bad, f"measurements straddling outer folds: {bad[:5]}"

    for k in range(N_OUTER):
        in_fold = df[df["outer_fold"] == k]
        assert in_fold["class_id"].nunique() == N_CLASSES, (
            f"outer fold {k} is missing classes: "
            f"{sorted(set(range(N_CLASSES)) - set(in_fold['class_id']))}"
        )

        col = f"inner_fold_for_outer_{k}"
        assert (in_fold[col] == -1).all(), (
            f"outer-test segments carry an inner assignment for fold {k}"
        )
        train = df[df["outer_fold"] != k]
        assert set(train[col]) == set(range(N_INNER)), (
            f"inner folds incomplete for outer fold {k}"
        )
        inner_spans = train.groupby("measurement_id")[col].nunique()
        bad_inner = inner_spans[inner_spans > 1].index.tolist()
        assert not bad_inner, (
            f"measurements straddling inner folds in outer {k}: {bad_inner[:5]}"
        )

    return {
        "n_segments": int(len(df)),
        "n_measurements": int(df["measurement_id"].nunique()),
        "n_edge_flagged": int(df["edge_flag"].sum()),
        "segments_per_class": df["class_name"].value_counts().to_dict(),
        "measurements_per_class": df.groupby("class_name")["measurement_id"]
        .nunique()
        .to_dict(),
        "raw_labels": sorted(df["label_raw"].unique().tolist()),
        "author_split_is_measurement_disjoint": bool(
            (df.groupby("measurement_id")["author_split"].nunique() == 1).all()
        ),
    }


def main() -> None:
    print(banner(), "\n")
    frozen = ARTIFACT_DIR / "fold_manifest.csv"
    if frozen.exists() and os.getenv("ALLOW_REFREEZE") != "1":
        raise SystemExit(
            f"{frozen} already exists.\n"
            "The fold manifest is the one artefact all four tracks must share "
            "byte for byte. Regenerating it on a different machine can produce "
            "different folds if scikit-learn versions differ, which silently "
            "breaks comparability.\n"
            "Generate it once, commit it, and have everyone read it. If you "
            "really must regenerate, set ALLOW_REFREEZE=1."
        )
    df = build_records()
    df = assign_folds(df)
    report = verify(df)

    out = ARTIFACT_DIR
    df.to_csv(out / "dataset_manifest.csv", index=False)
    df.to_csv(out / "fold_manifest.csv", index=False)

    fold_sha = hashlib.sha256(
        (out / "fold_manifest.csv").read_bytes()
    ).hexdigest()
    (out / "manifest_meta.json").write_text(
        json.dumps(
            {
                "n_rows": int(len(df)),
                "memmap_shape": [int(len(df)), N_RANGE, N_DOPPLER],
                "memmap_dtype": MEMMAP_DTYPE,
                "seed": SEED,
                "fold_manifest_sha256": fold_sha,
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                **report,
            },
            indent=2,
        )
    )
    print(f"\nfold_manifest.csv sha256: {fold_sha}")
    print("Every student must see this same hash. If it differs, the folds "
          "differ and results are not comparable.")

    print(json.dumps(report, indent=2))
    for label, got, want in (
        ("measurements", report["n_measurements"], EXPECTED_MEASUREMENTS),
        ("segments", report["n_segments"], EXPECTED_SEGMENTS),
        ("edge-flagged", report["n_edge_flagged"], EXPECTED_EDGE_FLAGGED),
    ):
        flag = "ok" if got == want else f"MISMATCH (ReadMe says {want})"
        print(f"  {label}: {got}  {flag}")
    print("\nAll fold assertions passed.")


if __name__ == "__main__":
    main()
