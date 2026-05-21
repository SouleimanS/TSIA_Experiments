#!/bin/bash
#PBS -N eval_v2_nb2
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o eval_v2_nb2.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

python -u scripts/eval_avhbench.py \
    --variant v2 \
    --ckpt-dir runs/avqa_v2_3ep_nb2 \
    --out-name v2_nb2

echo "=== Done: $(date) ==="
