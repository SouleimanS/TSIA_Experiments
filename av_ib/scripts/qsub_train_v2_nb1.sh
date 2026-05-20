#!/bin/bash
#PBS -N train_v2_nb1
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=06:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o train_v2_nb1.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

python -u scripts/train_avqa_v2_3ep.py --n-blocks 1 --output-dir runs/avqa_v2_3ep_nb1

echo "=== Done: $(date) ==="
