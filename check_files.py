"""
check_files.py -- confirm every S1 script on this machine is the current one.

Run from the project root:
    python check_files.py

Each entry lists one or more marker strings that exist only in the current
revision of that file; every marker must be present. This exists because a
stale copy of llm_verify.py silently produced two false-positive failure rates
that were reported as findings before anyone noticed the file had never been
replaced.

Markers now include the command-line flags that atlas_s1.sh actually passes.
A single behavioural marker was not enough: on 17 September the Atlas copy of
train_classical.py contained make_calibrated_svm and so was reported current,
yet predated --latency-repeats, and the batch job died with argparse exit
code 2 nineteen seconds in, after staging, preprocessing and feature
extraction had all succeeded. Any flag a runner passes is part of the
interface and is checked here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMON = ROOT / "src" / "common"
S1 = ROOT / "src" / "s1_physics"

# path -> (marker or tuple of markers that must all be present,
#          what that revision added)
CHECKS = {
    COMMON / "config.py": (
        ("ATLAS_ARTIFACT_DIR", "cannot create"),
        "three-way path split, ATLAS_* overrides, best-effort mkdir"),
    COMMON / "generate_manifest.py": (
        "fold_manifest_sha256",
        "write-once fold manifest with checksum and assertions"),
    COMMON / "preprocess_iq.py": (
        "linear_power",
        "uncentred linear power, no log, no min-max, memmapped output"),
    COMMON / "corruption.py": (
        "segment_rng",
        "per-segment deterministic seeding, signal-domain noise, blur"),
    S1 / "feature_extractor.py": (
        "ZERO_DOPPLER_HALFWIDTH",
        "power-weighted moments, CDF bandwidth, peak-anchored side lobe"),
    S1 / "feature_transforms.py": (
        ("signed_log1p", "COMPACT_SUBSET_IMPORTANCE"),
        "fold-scoped preprocessing with heavy-tail compression"),
    S1 / "validate_features.py": (
        "ZERO_DOPPLER_HALFWIDTH",
        "velocity and class name derived from the actual constant"),
    S1 / "zero_doppler_sweep.py": (
        "pairwise_auc",
        "band sweep with rank-based separability"),
    S1 / "train_classical.py": (
        ("make_calibrated_svm", "--latency-repeats", "--probe"),
        "grouped calibration, OOF predictions, model saving, repeated "
        "latency measurement"),
    S1 / "train_mlp.py": (
        ("_stratified_sample", "--demo"),
        "reproducibility demo fixed to use all four classes"),
    S1 / "analyze_results.py": (
        "drone_split",
        "per-class, confusion, worst measurements, drone quartiles"),
    S1 / "compare_models.py": (
        "measurement_recall",
        "paper tables plus consensus-failure analysis"),
    S1 / "run_robustness.py": (
        ("feature_stability", "_load_predictor"),
        "nine Section 17 conditions, stability, clean vs corrupted importance"),
    S1 / "run_ablations.py": (
        ("subsample_measurements", "compact5_importance", "partial_run"),
        "ablations, data efficiency, permutation importance"),
    S1 / "llm_verify.py": (
        "PROMPT_ECHO",
        "prompt-echo category so instruction numbers are not fabrications"),
    S1 / "llm_experiment.py": (
        "record_coverage",
        "model in filename, word count and record coverage, Wilson CIs"),
    S1 / "make_reliability.py": (
        "reliability_bins",
        "15-bin reliability diagrams and pooled ECE from stored probabilities"),
    S1 / "mlp_data_efficiency.py": (
        ("subsample_measurements", "selected_params_per_fold"),
        "MLP data-efficiency curve reusing the per-fold selected configuration"),
    S1 / "measure_mlp_latency.py": (
        "torch.cuda.synchronize",
        "batch-one MLP latency from the saved checkpoints"),
    S1 / "test_llm_verify.py": (
        "OLLAMA_RUN1",
        "27 tests including seven real-output regressions"),
}


def main() -> int:
    stale, missing, ok = [], [], []
    for path, (markers, why) in CHECKS.items():
        rel = path.relative_to(ROOT)
        if isinstance(markers, str):
            markers = (markers,)
        if not path.exists():
            missing.append((rel, why))
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            absent = [m for m in markers if m not in text]
            if absent:
                stale.append((rel, f"{why} [markers not found: "
                                   f"{', '.join(absent)}]"))
            else:
                ok.append(rel)

    for rel in ok:
        print(f"  current  {rel}")
    for rel, why in stale:
        print(f"  STALE    {rel}\n             missing: {why}")
    for rel, why in missing:
        print(f"  ABSENT   {rel}\n             would add: {why}")

    print(f"\n{len(ok)} current, {len(stale)} stale, {len(missing)} absent")
    if stale or missing:
        print("Replace the flagged files before running anything whose "
              "numbers will reach the paper.")
    return 1 if (stale or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
