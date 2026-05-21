#!/bin/bash
#PBS -N eval_musicavqa
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=01:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o eval_musicavqa.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"

echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

VARIANT="${VARIANT:-v3}"
CKPT_DIR="${CKPT_DIR:-runs/avqa_v3_3ep_b1e-2}"
OUT_NAME="${OUT_NAME:-v3_b1e-2_smoke}"
SPLIT="${SPLIT:-val}"
MAX_ITEMS="${MAX_ITEMS:-30}"
VIDEO_ROOT="${VIDEO_ROOT:-/home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all}"

echo "=== MUSIC-AVQA eval ==="
echo "  variant=$VARIANT  ckpt=$CKPT_DIR  out=$OUT_NAME"
echo "  split=$SPLIT  max_items=$MAX_ITEMS"
echo "  video_root=$VIDEO_ROOT"
echo ""

CMD=(python -u scripts/eval_musicavqa.py
    --variant "$VARIANT"
    --ckpt-dir "$CKPT_DIR"
    --out-name "$OUT_NAME"
    --split "$SPLIT"
    --video-root "$VIDEO_ROOT"
)
if [ "$MAX_ITEMS" != "0" ] && [ -n "$MAX_ITEMS" ]; then
    CMD+=(--max-items "$MAX_ITEMS")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=== Done: $(date) ==="
