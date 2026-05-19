#!/bin/bash
#PBS -N avqa_v3_3ep
#PBS -l walltime=06:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o train_avqa_v3_3ep.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0))"
echo ""

echo "=== Starting v3 3-epoch training (VIB, beta=1e-3) ==="
python -u scripts/train_avqa_v3_3ep.py
