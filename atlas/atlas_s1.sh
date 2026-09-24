#!/bin/bash
#SBATCH --job-name=se801-s1
#SBATCH --nodelist=argon01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/home/gmustafa/ANN_Project/logs/job_%j.log
#SBATCH --error=/home/gmustafa/ANN_Project/logs/job_%j.err

# ---------------------------------------------------------------------------
# SE-801 track S1 on Atlas. Everything that must carry an Atlas job ID or a
# comparable timing runs here, in ONE allocation on ONE node.
#
# Filesystems, per the Atlas wiki:
#   /home/gmustafa     scripts and logs. Shared with the compute nodes
#                      (verified), but NOT synced between the two login
#                      nodes, so always submit from raven01.
#   /data/gmustafa     networked and backed up. Dataset and every artefact
#                      that must outlive the job.
#   /scratch/gmustafa  node-local SSD, only reachable from inside a job. The
#                      dataset is staged here so reads come off local disk,
#                      and it is cleared at the end.
#
# Software: the module tree is not on MODULEPATH by default, hence the
# `module use`. pt_base supplies torch, numpy, scipy, pandas, scikit-learn,
# matplotlib, seaborn, psutil and joblib; ~/py_extra supplies xgboost and
# pytest, downloaded on the login node by atlas_setup.sh because the compute
# nodes cannot reach pypi.
#
# The node is pinned rather than left to the scheduler. Section 18 wants
# batch-one latency on one common node type, and Dr Farrukh warned the same
# setup times differently on repetition. Pinning argon01 removes node-to-node
# variation; --latency-repeats 5 measures what is left. The other three
# students should pin argon01 too, or the shared latency column compares
# nothing.
#
#   sbatch atlas_s1.sh
#   squeue -u $USER
#   tail -f /home/gmustafa/ANN_Project/logs/job_<id>.log
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT=/home/gmustafa/ANN_Project
DATASET=/data/gmustafa/se801/zenodo
ARTIFACTS=/data/gmustafa/se801/artifacts
SCRATCH=/scratch/gmustafa/se801
EXTRA=/home/gmustafa/py_extra

mkdir -p "$PROJECT/logs" "$ARTIFACTS" "$SCRATCH/zenodo" "$SCRATCH/work"

echo "=== job ${SLURM_JOB_ID:-none} on $(hostname) at $(date) ==="

# --- environment ----------------------------------------------------------
# `module` is a shell function from the login shell's init files, absent in a
# non-interactive batch shell, so source Lmod explicitly before using it.
if ! type module >/dev/null 2>&1; then
  for _init in /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh \
               /etc/profile.d/modules.sh; do
    [ -f "$_init" ] && { . "$_init"; break; }
  done
fi
type module >/dev/null 2>&1 || { echo "ERROR: Lmod not found" >&2; exit 1; }

module use /opt/atlas/modulefiles
module load 3.12/pt_base
export PYTHONPATH="$EXTRA:${PYTHONPATH:-}"

export PROJECT_ENV=atlas
export ATLAS_DATA_DIR="$SCRATCH/zenodo"
export ATLAS_WORK_DIR="$SCRATCH/work"
export ATLAS_ARTIFACT_DIR="$ARTIFACTS"

# joblib forks 16 workers; letting each spawn 16 BLAS threads is slower and
# makes the latency numbers noisy.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python3 -c "import torch,sklearn,numpy; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('sklearn',sklearn.__version__,'numpy',numpy.__version__)"

# --- stage the dataset onto the node's local SSD --------------------------
echo "=== staging dataset ==="
cp "$DATASET"/raw_iq_measurement_*.npy "$SCRATCH/zenodo/"
echo "staged $(ls "$SCRATCH/zenodo"/*.npy | wc -l) measurement files"

cd "$PROJECT"
python3 src/common/config.py
python3 check_files.py

# generate_manifest.py is deliberately NOT run. The fold manifest is
# write-once, its sha256 must match the copy the other three tracks use, and
# this node's scikit-learn (1.8.0) differs from the machine that produced it,
# so regenerating could silently change the folds.
for f in dataset_manifest.csv fold_manifest.csv manifest_meta.json; do
  [ -f "$ARTIFACTS/$f" ] || { echo "ERROR: $ARTIFACTS/$f missing. Transfer the data package first." >&2; exit 1; }
done
echo "fold_manifest.csv sha256: $(sha256sum "$ARTIFACTS/fold_manifest.csv" | cut -d' ' -f1)"

echo "=== preprocessing and features ==="
python3 src/common/preprocess_iq.py
python3 src/s1_physics/feature_extractor.py
python3 src/s1_physics/validate_features.py

echo "=== classical models, latency on this node ==="
python3 src/s1_physics/train_classical.py \
    --models svm rf xgb \
    --n-jobs "${SLURM_CPUS_PER_TASK:-16}" \
    --latency-repeats 5

echo "=== MLP, with both section 14 demonstrations ==="
python3 src/s1_physics/train_mlp.py --demo overfit
python3 src/s1_physics/train_mlp.py --demo repro
python3 src/s1_physics/train_mlp.py

# --- clean up -------------------------------------------------------------
# The wiki asks users to clear scratch. Everything here was staged from /data
# or is regenerable, so nothing is lost.
echo "=== clearing scratch ==="
rm -rf "$SCRATCH"

echo "=== done, job ${SLURM_JOB_ID:-none} at $(date) ==="
echo "artefacts: $ARTIFACTS"
