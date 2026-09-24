#!/bin/bash
#SBATCH --job-name=s1-mlp-latency
#SBATCH --nodelist=argon01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/gmustafa/ANN_Project/logs/latency_%j.log
#SBATCH --error=/home/gmustafa/ANN_Project/logs/latency_%j.err
#
# Measures the MLP's batch-one inference latency from the checkpoints job 2305
# already wrote, closing the last Section 18 gap without retraining.
#
#   sbatch atlas_latency.sh
#
# This is a real bash script with a shebang, unlike `sbatch --wrap`, which
# builds a /bin/sh script. On Atlas that distinction matters: `module` is a
# shell function defined by the Lmod init file, and neither `source` nor that
# function exists in dash, so a wrapped one-liner fails in one second with
# "source: not found" followed by ModuleNotFoundError.
#
# Node pinned to argon01 so the figure is comparable with the classical
# latencies from job 2305, which were measured on the same node.

set -euo pipefail

PROJ=/home/gmustafa/ANN_Project
cd "$PROJ"

echo "=== $(date) on $(hostname) ==="

# Lmod lives only on the compute nodes and is not on MODULEPATH by default.
source /usr/share/lmod/lmod/init/bash
module use /opt/atlas/modulefiles
module load 3.12/pt_base

# xgboost and pytest were pip-installed on the login node into ~/py_extra,
# because the compute nodes cannot reach pypi.
export PYTHONPATH="$HOME/py_extra${PYTHONPATH:+:$PYTHONPATH}"
export PROJECT_ENV=atlas

# One thread per process. Batch-one latency on a machine that is quietly
# running BLAS threads in the background is not batch-one latency.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "--- environment ---"
python3 - <<'PY'
import torch, numpy, pandas, sklearn
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("numpy", numpy.__version__, "pandas", pandas.__version__,
      "sklearn", sklearn.__version__)
PY

echo "--- checkpoints and features present? ---"
ls -1 /data/gmustafa/se801/artifacts/models/mlp_fold*.pt
ls -l /data/gmustafa/se801/artifacts/s1_features.csv

echo "--- batch-one latency, GPU and CPU ---"
python3 src/s1_physics/measure_mlp_latency.py --also-cpu

echo "=== done $(date) ==="
echo "results: /data/gmustafa/se801/artifacts/results_mlp_latency.json"
