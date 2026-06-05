#!/usr/bin/env bash
# Create the organocat Python environment ON A LOGIN NODE (needs internet).
# Caltech compute nodes typically have NO internet, so all installs happen here.
#
#   bash setup_env.sh
#
# This is the step most likely to need tweaking on a new cluster. It is verbose
# and stops on first error so failures are easy to relay.
set -euo pipefail

ENV_NAME="${ENV_NAME:-organocat}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[setup] repo root: $ROOT"

# ---- 1. obtain a conda ------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    echo "[setup] using existing conda: $(command -v conda)"
    CONDA_BASE="$(conda info --base)"
else
    echo "[setup] no conda found; installing Miniforge into \$ROOT/../miniforge3"
    MF="$ROOT/../miniforge3"
    if [ ! -d "$MF" ]; then
        URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        wget -q "$URL" -O /tmp/mf.sh || curl -fsSL "$URL" -o /tmp/mf.sh
        bash /tmp/mf.sh -b -p "$MF"
    fi
    CONDA_BASE="$MF"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- 2. create env ----------------------------------------------------------
if ! conda env list | grep -qE "^\s*$ENV_NAME\s"; then
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
python -V

# git-lfs is absent on this cluster; install into the env (needed by download_ord.sh)
if ! command -v git-lfs >/dev/null 2>&1; then
    conda install -y -c conda-forge git-lfs
fi
git lfs install

# ---- 3. install deps (order matters: torch first) ---------------------------
# torch ships its own CUDA runtime; the default linux wheel is CUDA-enabled and
# works on most modern HPC GPUs regardless of the system CUDA module.
pip install --upgrade pip
pip install torch
pip install rxnmapper            # resolves a compatible transformers/tokenizers
pip install rdkit                # ensure a recent rdkit (rxnmapper may pin old)
pip install torch_geometric
pip install ord-schema pandas pyarrow scikit-learn numpy

echo
echo "[setup] verifying imports ..."
python - <<'PY'
import torch, torch_geometric, rdkit, sklearn, pandas, numpy
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
print("rdkit", rdkit.__version__)
try:
    from rxnmapper import RXNMapper; print("rxnmapper OK")
except Exception as e:
    print("rxnmapper IMPORT FAILED:", e)
try:
    from ord_schema.proto import reaction_pb2; print("ord_schema OK")
except Exception as e:
    print("ord_schema IMPORT FAILED:", e)
PY
echo
echo "[setup] done. Activate with:  conda activate $ENV_NAME"
