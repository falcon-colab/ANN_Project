#!/bin/bash
#SBATCH --job-name=s1-ablate
#SBATCH --nodelist=argon01
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/home/gmustafa/ANN_Project/logs/ablate_%j.log
#SBATCH --error=/home/gmustafa/ANN_Project/logs/ablate_%j.err
#
# Ablations, data efficiency and robustness on Atlas, so that every classical
# number in the paper comes from one machine.
#
#   sbatch atlas_ablations.sh
#
# No GPU is requested: all three analyses are CPU only. They reuse the
# hyperparameters and the fitted models that job 2305 wrote to /data, so
# nothing is retuned and nothing is refitted on corrupted data.
#
# Robustness recomputes every descriptor from corrupted I/Q, so the raw
# measurements have to be staged to the node's SSD exactly as in atlas_s1.sh.
# Ablations only need s1_features.csv, which is already in /data.
#
# generate_manifest.py is never run. The fold manifest is write-once and its
# sha256 must stay identical to the copy the other three tracks use.

set -euo pipefail

PROJECT=/home/gmustafa/ANN_Project
DATASET=/data/gmustafa/se801/zenodo
ARTIFACTS=/data/gmustafa/se801/artifacts
SCRATCH=/scratch/gmustafa/se801
NJOBS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "$PROJECT/logs" "$ARTIFACTS" "$SCRATCH/zenodo" "$SCRATCH/work"

echo "=== job ${SLURM_JOB_ID:-none} on $(hostname) at $(date) ==="

# --- environment ----------------------------------------------------------
# Lmod exists only on the compute nodes and its modulefiles are not on
# MODULEPATH by default. `module` is a shell function, so the init file has to
# be sourced first, which is also why this is a bash script and not an
# sbatch --wrap one-liner.
source /usr/share/lmod/lmod/init/bash
module use /opt/atlas/modulefiles
module load 3.12/pt_base
export PYTHONPATH="$HOME/py_extra${PYTHONPATH:+:$PYTHONPATH}"

export PROJECT_ENV=atlas
export ATLAS_DATA_DIR="$SCRATCH/zenodo"
export ATLAS_WORK_DIR="$SCRATCH/work"
export ATLAS_ARTIFACT_DIR="$ARTIFACTS"

# joblib forks 16 workers; one BLAS thread each, or they oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$PROJECT"
python3 src/common/config.py
python3 check_files.py

echo "=== inputs from job 2305 that these analyses depend on ==="
ls -l "$ARTIFACTS/results_classical_full.json" "$ARTIFACTS/s1_features.csv"
ls -1 "$ARTIFACTS"/models/*.joblib | head -20
echo "fold manifest sha256:"
sha256sum "$ARTIFACTS/fold_manifest.csv"

# --- stage the raw measurements for the robustness stage -------------------
echo "=== staging dataset to $SCRATCH/zenodo ==="
cp "$DATASET"/raw_iq_measurement_*.npy "$SCRATCH/zenodo/"
echo "staged $(ls "$SCRATCH/zenodo"/*.npy | wc -l) measurement files"

# The memmap is regenerated on the node rather than copied over the network.
echo "=== preprocessing to node-local scratch ==="
python3 src/common/preprocess_iq.py

# --- the three analyses ---------------------------------------------------
# Ablations now report BOTH five-feature subsets: the one chosen from the
# correlation structure before any training, and the one chosen from
# permutation importance on the full-feature models. The second is not a blind
# choice and the report says so.
# One invocation, not three with --only. All three sections land in the same
# results file, and a second invocation would rewrite it; the script now merges
# when --only is used, but doing it in one pass is simpler and no slower.
echo "=== ablations, data efficiency and permutation importance ==="
python3 src/s1_physics/run_ablations.py \
    --models svm xgb rf --n-jobs "$NJOBS"

echo "=== robustness, nine section 17 conditions ==="
python3 src/s1_physics/run_robustness.py \
    --models svm xgb rf --n-jobs "$NJOBS"

# --- clean up the node ----------------------------------------------------
echo "=== clearing scratch ==="
rm -rf "$SCRATCH"

echo "=== done, job ${SLURM_JOB_ID:-none} at $(date) ==="
echo "artefacts are in $ARTIFACTS"
