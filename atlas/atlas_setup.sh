#!/bin/bash
# atlas_setup.sh -- run ONCE on the raven01 LOGIN node.
#
#   bash atlas_setup.sh
#
# Why this runs on the login node and not in a job: raven01 has working
# internet, and the argon compute nodes do not (their curl fails against the
# conda environment's own libcurl). So every download happens here.
#
# Why a plain directory instead of a virtual environment: the Atlas software
# comes from the 3.12/pt_base module, which exists only on the compute nodes.
# A venv built here could not see it, and a venv built there could not reach
# pypi. Installing the two missing packages into a directory and putting that
# directory on PYTHONPATH avoids both problems. pt_base is Python 3.12 and so
# is raven01, so the wheels match.
#
# pt_base already provides torch, numpy, scipy, pandas, scikit-learn,
# matplotlib, seaborn, psutil and joblib. Only xgboost and pytest are missing.

set -euo pipefail

PROJECT="$HOME/ANN_Project"
EXTRA="$HOME/py_extra"

echo "login node: $(hostname)"
echo "python:     $(python3 -V)"
echo

# --- pip, one way or another ----------------------------------------------
# The system Python may have no pip. A throwaway venv always provides one.
if python3 -m pip --version >/dev/null 2>&1; then
  PIP=(python3 -m pip)
  echo "using system pip"
else
  echo "system pip absent; creating a throwaway venv for downloading"
  rm -rf /tmp/_dlvenv
  python3 -m venv /tmp/_dlvenv
  PIP=(/tmp/_dlvenv/bin/pip)
fi

echo
echo "=== installing xgboost and pytest into $EXTRA ==="
rm -rf "$EXTRA"
mkdir -p "$EXTRA"
"${PIP[@]}" install --upgrade pip >/dev/null 2>&1 || true
"${PIP[@]}" install --target "$EXTRA" xgboost pytest

echo
echo "=== directories ==="
mkdir -p /data/"$USER"/se801/zenodo /data/"$USER"/se801/artifacts "$PROJECT/logs"
ls -ld /data/"$USER"/se801/zenodo /data/"$USER"/se801/artifacts "$PROJECT/logs"

echo
echo "=== verifying on a compute node, where the job will actually run ==="
srun --nodelist=argon01 --gres=gpu:1 bash -lc '
  module use /opt/atlas/modulefiles
  module load 3.12/pt_base
  export PYTHONPATH="'"$EXTRA"':${PYTHONPATH:-}"
  echo "  node: $(hostname)"
  python3 - <<PY
import importlib, sys
need = ["numpy","scipy","pandas","sklearn","xgboost","joblib",
        "matplotlib","seaborn","psutil","torch"]
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:<12} {getattr(mod,'__version__','')}")
    except ImportError as e:
        missing.append(m); print(f"  MISS {m:<12} {e}")
import torch
print(f"\n  torch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
import sklearn, numpy
print(f"\n  scikit-learn {sklearn.__version__} | numpy {numpy.__version__}")
print("  NOTE: these decide feature values. If they differ from the home PC,")
print("  say so in the paper and never regenerate fold_manifest.csv here.")
sys.exit(1 if missing else 0)
PY
'

echo
echo "Setup complete. Next: transfer the dataset, then  sbatch atlas_s1.sh"
