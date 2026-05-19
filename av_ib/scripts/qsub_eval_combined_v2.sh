#!/bin/bash
#PBS -N eval_comb_v2
#PBS -l walltime=01:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o eval_combined_v2.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

echo "=== Evaluating combined-trained v2 on held-out AVHBench split ==="
python -u scripts/eval_avhbench.py \
  --variant v2 \
  --ckpt-dir runs/combined_v2_5ep \
  --test-file data/avhbench_split_test.json \
  --out-name combined_v2
