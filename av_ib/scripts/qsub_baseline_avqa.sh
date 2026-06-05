#!/bin/bash
#PBS -N baseline_avqa
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=96:00:00
#PBS -j oe
#PBS -o baseline_avqa.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/AVQA/AVQA/AVQA_dataset/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/AVQA/videos/Train"

mkdir -p runs/baseline_avqa

echo "=== Qwen3-Omni LoRA baseline | dataset=avqa ==="
python -u -m av_ib.train.train_baseline \
    --dataset avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --num-steps 63854 \
    --lr 1e-4 \
    --log-path runs/baseline_avqa/log.jsonl \
    --ckpt-path runs/baseline_avqa/final.pt \
    --save-every 2000 \
    --print-every 100

echo "=== Done $(date) ==="
