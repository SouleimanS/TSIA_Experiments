#!/bin/bash
#PBS -N avqa_v1_3ep_ddp
#PBS -l walltime=03:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o train_avqa_3ep_ddp.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"
echo ""

export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Launching torchrun on 8 GPUs ==="
torchrun --standalone --nproc_per_node=8 scripts/train_avqa_3ep_ddp.py
