#!/bin/bash
#PBS -N eval_best_full
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -j oe
#PBS -o eval_best_full.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
python -u scripts/eval_best_full_val.py 2>&1
