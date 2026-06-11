#!/bin/bash
#PBS -N avqa_lora
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o avqa_lora.qsub.log

# Condition 5: vanilla Qwen3-Omni + LoRA on AVQA (no bottleneck) — the
# uncompressed reference point of the rate-distortion comparison.

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/avqa_lora_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/avqa_lora"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/AVQA/train_qa.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/AVQA/videos"
[ -f "$ANN_PATH" ] || { echo "ERROR: $ANN_PATH missing — run download_avqa.py" >&2; exit 1; }

python -u -m av_ib.train.train_baseline \
    --dataset avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --num-steps 2000 \
    --lr 1e-4 \
    --log-path runs/avqa_lora/log.jsonl \
    --ckpt-path runs/avqa_lora/final.pt \
    --save-every 500 \
    --print-every 50

echo "=== Done $(date) ==="
