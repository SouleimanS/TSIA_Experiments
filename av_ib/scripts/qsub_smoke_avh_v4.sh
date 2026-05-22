#!/bin/bash
#PBS -N smoke_avh_v4
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=00:20:00
#PBS -l select=1
#PBS -j oe
#PBS -o smoke_avh_v4.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
python -u scripts/eval_avhbench.py \
    --variant v4 \
    --ckpt-dir runs/musicavqa_v4_3ep_b1e-2 \
    --out-name smoke_v4 \
    --max-items 20
