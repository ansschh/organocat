#!/usr/bin/env bash
#SBATCH --job-name=oeg-smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.out
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu
#SBATCH --account=tensorlab
#
# Fast end-to-end validation of the bench on the H100: small slice, 2 epochs,
# one split. Confirms train -> eval -> baselines -> decision run clean before
# the full 3-split / 15-epoch job.
set -euo pipefail
export ORGANOCAT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORGANOCAT_ROOT"
mkdir -p logs
export PYTHONPATH="$ORGANOCAT_ROOT:${PYTHONPATH:-}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-organocat}"

python scripts/run_zcat_bench.py \
    --limit 12000 --epochs 2 --splits leave_catalyst_out \
    --batch-size 64 --embed-dim 128 \
    --out-json data/smoke_bench.json
echo "=== smoke complete ==="
