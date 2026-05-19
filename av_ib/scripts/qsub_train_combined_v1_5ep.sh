#!/bin/bash
#PBS -N comb_v1_5ep
#PBS -l walltime=08:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o train_combined_v1_5ep.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0))"
echo ""

echo "=== Starting combined v1 5-epoch training ==="
python -u scripts/train_combined_v1_5ep.py
