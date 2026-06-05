#!/bin/bash
#PBS -N baseline_mavqa
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=96:00:00
#PBS -j oe
#PBS -o baseline_mavqa.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/baseline_mavqa

echo "=== Qwen3-Omni LoRA baseline | dataset=music_avqa ==="
python -u -m av_ib.train.train_baseline \
    --dataset music_avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --num-steps 63854 \
    --lr 1e-4 \
    --log-path runs/baseline_mavqa/log.jsonl \
    --ckpt-path runs/baseline_mavqa/final.pt \
    --save-every 2000 \
    --print-every 100

echo "=== Done $(date) ==="
