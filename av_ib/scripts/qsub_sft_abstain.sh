#!/bin/bash
#PBS -N sft_abstain
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o sft_abstain.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/sft_abstain_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/sft"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

LABELS="$PBS_O_WORKDIR/runs/abstain_labels.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

# GATE: submit only after Stage-0 shows the abstain prompt alone does NOT
# already fail-safe, and the miner produced abstain_labels.json with enough
# vid-flip contrast pairs.
# SFT head-to-head with DPO (same data, same conditions): FINER reports DPO>SFT;
# we verify that on our setup.

if [ ! -f "$LABELS" ]; then
    echo "ERROR: $LABELS not found — run abstain_labels miner first." >&2
    exit 1
fi

# ── C2a-SFT: b_video_only VIB-only (frozen LLM) ──
echo "=== C2a-SFT | b_video_only | VIB-only (no LoRA) ==="
python -u -m av_ib.train.train_sft_abstain \
    --labels-json "$LABELS" --video-root "$VIDEO_ROOT" \
    --variant b_video_only \
    --num-steps 400 --lr 5e-5 --anchor-ratio 1.0 \
    --lam-abstain 1.0 --lam-anchor 0.5 --lam-negctrl 0.5 \
    --log-path runs/sft/c2a_sft.jsonl \
    --ckpt-path runs/sft/c2a_sft_final.pt

# ── C1-SFT: vanilla + LoRA (VIB frozen) ──
echo "=== C1-SFT | vanilla + LoRA | VIB frozen ==="
python -u -m av_ib.train.train_sft_abstain \
    --labels-json "$LABELS" --video-root "$VIDEO_ROOT" \
    --variant b_video_only --use-lora --freeze-vib \
    --num-steps 400 --lr 5e-5 --anchor-ratio 1.0 \
    --lam-abstain 1.0 --lam-anchor 0.5 --lam-negctrl 0.5 \
    --log-path runs/sft/c1_sft.jsonl \
    --ckpt-path runs/sft/c1_sft_final.pt

# ── C2b-SFT: b_video_only + LoRA ──
echo "=== C2b-SFT | b_video_only + LoRA ==="
python -u -m av_ib.train.train_sft_abstain \
    --labels-json "$LABELS" --video-root "$VIDEO_ROOT" \
    --variant b_video_only --use-lora \
    --num-steps 400 --lr 5e-5 --anchor-ratio 1.0 \
    --lam-abstain 1.0 --lam-anchor 0.5 --lam-negctrl 0.5 \
    --log-path runs/sft/c2b_sft.jsonl \
    --ckpt-path runs/sft/c2b_sft_final.pt

echo "=== Done $(date) ==="
