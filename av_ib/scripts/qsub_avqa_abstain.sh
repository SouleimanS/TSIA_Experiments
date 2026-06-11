#!/bin/bash
#PBS -N avqa_abstain
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o avqa_abstain.qsub.log

# Conditions 1-3 of the AVQA experiment grid:
#   1. video bottleneck + DPO          (b_video_only)
#   2. video bottleneck + SFT          (b_video_only)
#   3. video+audio bottleneck + DPO    (b_topk_nofusion)
# PREREQUISITE: runs/abstain_labels_avqa.json from qsub_abstain_labels_avqa.sh.

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/avqa_abstain_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/avqa_abstain"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# No CUDA_VISIBLE_DEVICES pin: device_map="auto" shards across all GPUs.

LABELS="$PBS_O_WORKDIR/runs/abstain_labels_avqa.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/AVQA/videos"
[ -f "$LABELS" ] || { echo "ERROR: $LABELS missing — run qsub_abstain_labels_avqa.sh first." >&2; exit 1; }

# ── 1. video bottleneck + DPO ──
echo "=== AVQA | b_video_only | DPO ==="
python -u -m av_ib.train.train_dpo \
    --labels-json "$LABELS" --video-root "$VIDEO_ROOT" \
    --variant b_video_only \
    --num-steps 400 --beta-dpo 0.1 --lr 5e-5 --anchor-ratio 1.0 \
    --log-path runs/avqa_abstain/vib_dpo.jsonl \
    --ckpt-path runs/avqa_abstain/vib_dpo_final.pt

# ── 2. video bottleneck + SFT ──
echo "=== AVQA | b_video_only | SFT ==="
python -u -m av_ib.train.train_sft_abstain \
    --labels-json "$LABELS" --video-root "$VIDEO_ROOT" \
    --variant b_video_only \
    --num-steps 400 --lr 5e-5 \
    --log-path runs/avqa_abstain/vib_sft.jsonl \
    --ckpt-path runs/avqa_abstain/vib_sft_final.pt

# ── 3. video+audio bottleneck + DPO ──
echo "=== AVQA | b_topk_nofusion (video+audio VIB) | DPO ==="
python -u -m av_ib.train.train_dpo \
    --labels-json "$LABELS" --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --num-steps 400 --beta-dpo 0.1 --lr 5e-5 --anchor-ratio 1.0 \
    --log-path runs/avqa_abstain/avvib_dpo.jsonl \
    --ckpt-path runs/avqa_abstain/avvib_dpo_final.pt

echo "=== Done $(date) ==="
