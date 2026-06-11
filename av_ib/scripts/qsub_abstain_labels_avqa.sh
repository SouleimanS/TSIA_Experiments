#!/bin/bash
#PBS -N abstain_labels_avqa
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -o abstain_labels_avqa.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/abstain_labels_avqa_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0   # single GPU: VIB hooks need same-device tensors

# ── ADJUST THESE to where the AVQA annotations + VGGSound clips live ──
ANN_PATH="$HOME/SOULEIMAN_repo/datasets/AVQA/train_qa.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/AVQA/videos"

if [ ! -f "$ANN_PATH" ]; then
    echo "ERROR: $ANN_PATH not found — download AVQA annotations first." >&2
    exit 1
fi

# GATE: run qsub_explain_e1_avqa.sh first; only mine if AVQA shows genuine
# AV reliance (it should — open-domain VGGSound questions need listening).
echo "=== abstain-label mining on AVQA | untrained model | 3000 candidates ==="
python -u -m av_ib.eval.explain alabels \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --dataset avqa \
    --variant b_video_only \
    --num-samples 3000 --seed 42 --every 50 \
    --out-json runs/abstain_labels_avqa.json

echo "=== Done $(date) ==="
