#!/bin/bash
#PBS -N v5_train_sanity
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o v5_train_sanity.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/sanity_b0

echo "=== v5 sanity: 100 steps, beta=0, lr=1e-4 ==="
python -u -m av_ib.train.train_v5 \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --num-steps 100 \
    --lr 1e-4 \
    --beta-v 0 --beta-a 0 --beta-j 0 \
    --aux-weight 0.1 \
    --log-path runs/sanity_b0/log.jsonl \
    --ckpt-path runs/sanity_b0/final.pt \
    --print-every 1

echo "=== Done $(date) ==="
