#!/bin/bash
#PBS -N pilot_baseline
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o pilot_baseline.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/qsub_pilot_baseline_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/pilot_baseline

echo "=== PILOT: Qwen3-Omni LoRA baseline | 2k samples seed=42 ==="
python -u -m av_ib.train.train_baseline \
    --dataset music_avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --max-samples 2000 --seed 42 \
    --num-steps 2000 \
    --lr 1e-4 \
    --log-path runs/pilot_baseline/log.jsonl \
    --ckpt-path runs/pilot_baseline/final.pt \
    --print-every 50

echo "=== Done $(date) ==="
