#!/bin/bash
#PBS -N train_mavqa_v1
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=12:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o train_musicavqa_v1.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
python -u scripts/train_musicavqa_v1_3ep.py --output-dir runs/musicavqa_v1_3ep
echo "=== Done: $(date) ==="
