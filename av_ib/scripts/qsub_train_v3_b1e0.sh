#!/bin/bash
#PBS -N train_v3_b1e0
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=06:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o train_v3_b1e0.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

python -u scripts/train_avqa_v3_3ep.py --beta 1.0  --output-dir runs/avqa_v3_3ep_b1e0

echo "=== Done: $(date) ==="
