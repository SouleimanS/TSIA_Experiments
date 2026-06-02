#!/bin/bash
#PBS -N v6_tier1_c
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o v6_tier1_c.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/v6_tier1_c

echo "=== v6 Tier 1 variant c: 1000 steps, beta=0, lr=1e-4, save every 100 ==="
python -u -m av_ib.train.train_v6 \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --num-steps 1000 \
    --lr 1e-4 \
    --beta-v 0 --beta-a 0 --beta-j 0 \
    --aux-weight 0.1 \
    --log-path runs/v6_tier1_c/log.jsonl \
    --ckpt-path runs/v6_tier1_c/final.pt \
    --save-every 10 \
    --print-every 50 \
    --save-every 100 \
    --variant c

echo "=== Done $(date) ==="
