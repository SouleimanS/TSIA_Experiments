#!/bin/bash
#PBS -N dpo_abstain
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o dpo_abstain.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/dpo_abstain_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/dpo"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# No CUDA_VISIBLE_DEVICES pin: let device_map="auto" shard the 31.7B model
# across all 8 H200s (training needs the headroom; the C-MIB provider migrates
# VIB tensors to the encoder-output device, so cross-GPU sharding is fine).

LABELS="$PBS_O_WORKDIR/runs/abstain_labels.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

# GATE: only run after (1) Stage-0 shows the abstain prompt alone does NOT
# already fail-safe, and (2) the miner produced abstain_labels.json with enough
# vid-flip (not aud-flip) contrast pairs. FINER-inspired DPO; see
# docs/abstention_dpo_plan.md.

if [ ! -f "$LABELS" ]; then
    echo "ERROR: $LABELS not found — run abstain_labels miner first." >&2
    exit 1
fi

# ── C2a-DPO: b_video_only VIB-only (frozen LLM) + DPO ──
echo "=== C2a-DPO | b_video_only | VIB-only (no LoRA) | beta_dpo=0.1 ==="
python -u -m av_ib.train.train_dpo \
    --labels-json "$LABELS" --video-root "$VIDEO_ROOT" \
    --variant b_video_only \
    --num-steps 400 --beta-dpo 0.1 --lr 5e-5 --anchor-ratio 1.0 \
    --log-path runs/dpo/c2a_dpo.jsonl \
    --ckpt-path runs/dpo/c2a_dpo_final.pt

# ── C1-DPO: vanilla + LoRA (VIB frozen) + DPO — the no-bottleneck baseline ──
echo "=== C1-DPO | vanilla + LoRA | VIB frozen | beta_dpo=0.1 ==="
python -u -m av_ib.train.train_dpo \
    --labels-json "$LABELS" --video-root "$VIDEO_ROOT" \
    --variant b_video_only --use-lora --freeze-vib \
    --num-steps 400 --beta-dpo 0.1 --lr 5e-5 --anchor-ratio 1.0 \
    --log-path runs/dpo/c1_dpo.jsonl \
    --ckpt-path runs/dpo/c1_dpo_final.pt

# ── C2b-DPO: b_video_only + LoRA + DPO ──
echo "=== C2b-DPO | b_video_only + LoRA | beta_dpo=0.1 ==="
python -u -m av_ib.train.train_dpo \
    --labels-json "$LABELS" --video-root "$VIDEO_ROOT" \
    --variant b_video_only --use-lora \
    --num-steps 400 --beta-dpo 0.1 --lr 5e-5 --anchor-ratio 1.0 \
    --log-path runs/dpo/c2b_dpo.jsonl \
    --ckpt-path runs/dpo/c2b_dpo_final.pt

echo "=== Done $(date) ==="
