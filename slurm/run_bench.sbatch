#!/usr/bin/env bash
#SBATCH --job-name=oeg-bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# >>> EDIT for your allocation <<<
#SBATCH --partition=gpu
##SBATCH --account=CHANGE_ME
#
# RAS-Bench v0: trains Z_cat-CLIP across splits and prints the go/no-go table.
set -euo pipefail

export ORGANOCAT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ORGANOCAT_ROOT"
mkdir -p logs
export PYTHONPATH="$ORGANOCAT_ROOT:${PYTHONPATH:-}"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-organocat}"

python scripts/run_zcat_bench.py \
    --pairs-pkl data/pairs/contrastive_pairs.pt \
    --clean-pkl data/reactions/ord_tm_clean.pkl \
    --splits leave_catalyst_out leave_metal_out leave_ligand_out \
    --epochs 15 --batch-size 64 --embed-dim 128 \
    --out-json data/ras_bench_v0.json

echo "=== bench complete -> data/ras_bench_v0.json ==="
