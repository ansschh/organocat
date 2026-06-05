#!/usr/bin/env bash
#SBATCH --job-name=oeg-build
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.out
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
# Caltech Resnick HPC (finalized from `sinfo`/`sacctmgr`):
#SBATCH --partition=gpu
#SBATCH --account=tensorlab
#
# Builds the dataset from raw ORD:
#   parse_ord -> clean_and_atom_map_ord -> contrastive_dataset (3D pairs)
# Assumes setup_env.sh already ran on a login node and ORD was downloaded.
set -euo pipefail

# SLURM copies this script to a spool dir, so $0 is NOT the repo path.
# Use the directory sbatch was invoked from (the repo root).
export ORGANOCAT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORGANOCAT_ROOT"
mkdir -p logs data/reactions data/pairs
export PYTHONPATH="$ORGANOCAT_ROOT:${PYTHONPATH:-}"

# activate conda env
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-organocat}"

echo "=== [1/3] parse ORD -> TM-tagged reactions ==="
python scripts/parse_ord.py \
    --ord-root data/raw/ord-data/data \
    --out data/reactions/ord_tm_reactions.pkl

echo "=== [2/3] clean + atom-map (rxnmapper) ==="
python scripts/clean_and_atom_map_ord.py \
    --in-pkl data/reactions/ord_tm_reactions.pkl \
    --out-pkl data/reactions/ord_tm_clean.pkl \
    --batch-size 64

echo "=== [3/3] build 3D contrastive pairs (ETKDG, per-mol timeout) ==="
python src/data/contrastive_dataset.py \
    --clean-pkl data/reactions/ord_tm_clean.pkl \
    --out-pkl data/pairs/contrastive_pairs.pt \
    --cat-cache-pkl data/pairs/cat_cache.pkl \
    --timeout-s 15

echo "=== build complete ==="
ls -la data/reactions data/pairs
