#!/bin/bash
#PBS -N v6_bv_0p001
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o v6_betav_0p001.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/v6_betav_0p001

echo "=== v6 variant b, beta_v=0.001, 1000 steps ==="
python -u -m av_ib.train.train_v6 \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --num-steps 1000 \
    --lr 1e-4 \
    --beta-v 0.001 --beta-a 0 --beta-j 0 \
    --aux-weight 0.1 \
    --log-path runs/v6_betav_0p001/log.jsonl \
    --ckpt-path runs/v6_betav_0p001/final.pt \
    --save-every 100 \
    --print-every 50 \
    --variant b

echo "=== Done $(date) ==="
