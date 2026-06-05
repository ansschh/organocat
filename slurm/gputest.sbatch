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
python - <<'PY'
import torch
print('TORCH', torch.__version__, '| CUDA_AVAIL', torch.cuda.is_available())
if torch.cuda.is_available():
    dev = torch.device('cuda')
    try:
        x = torch.randn(128, 128, device=dev)
        val = float((x @ x).sum().item())   # forces a real kernel launch
        print('GPU_KERNEL_OK sum=%.3f | DEV %s' % (val, torch.cuda.get_device_name(0)))
    except Exception as e:
        print('GPU_KERNEL_FAILED:', repr(e))
else:
    print('NO-GPU')
PY
echo '--- nvidia-smi ---'
nvidia-smi || true
