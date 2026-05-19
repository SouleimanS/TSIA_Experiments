#!/bin/bash
#PBS -N eval_comb_v1
#PBS -l walltime=01:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o eval_combined_v1.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

echo "=== Evaluating combined-trained v1 on held-out AVHBench split ==="
python -u scripts/eval_avhbench.py \
  --variant v1 \
  --ckpt-dir runs/combined_v1_5ep \
  --test-file data/avhbench_split_test.json \
  --out-name combined_v1
