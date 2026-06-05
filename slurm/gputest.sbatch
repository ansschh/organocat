#!/usr/bin/env bash
#SBATCH --job-name=gputest
#SBATCH --partition=gpu
#SBATCH --account=tensorlab
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/gputest.out
# Confirms the cu130 torch build actually sees the GPU on a compute node.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate organocat
python -c "import torch; print('TORCH', torch.__version__, '| CUDA_AVAIL', torch.cuda.is_available(), '| DEV', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"
echo '--- nvidia-smi ---'
nvidia-smi || true
