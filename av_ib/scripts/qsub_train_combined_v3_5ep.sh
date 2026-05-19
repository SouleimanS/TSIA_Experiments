#!/bin/bash
#PBS -N comb_v3_5ep
#PBS -l walltime=08:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o train_combined_v3_5ep.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
echo "=== Starting combined v3 5-epoch training ==="
python -u scripts/train_combined_v3_5ep.py
