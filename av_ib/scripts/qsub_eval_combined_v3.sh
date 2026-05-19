#!/bin/bash
#PBS -N eval_comb_v3
#PBS -l walltime=01:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o eval_combined_v3.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

echo "=== Evaluating combined-trained v3 on held-out AVHBench split ==="
python -u scripts/eval_avhbench.py \
  --variant v3 \
  --ckpt-dir runs/combined_v3_5ep \
  --test-file data/avhbench_split_test.json \
  --out-name combined_v3
