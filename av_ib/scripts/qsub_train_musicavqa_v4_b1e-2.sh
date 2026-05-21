#!/bin/bash
#PBS -N train_mavqa_v4_b1e-2
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=14:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o train_musicavqa_v4_b1e-2.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
python -u scripts/train_musicavqa_v4_3ep.py \
    --beta-v 1e-2 --beta-a 1e-2 --beta-j 1e-2 \
    --aux-weight 1.0 \
    --output-dir runs/musicavqa_v4_3ep_b1e-2
echo "=== Done: $(date) ==="
