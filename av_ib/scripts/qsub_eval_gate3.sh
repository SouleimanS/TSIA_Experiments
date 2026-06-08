#!/bin/bash
#PBS -N eval_gate3
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o eval_gate3.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/eval_gate3_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

COMMON="--ann-path $ANN_PATH --video-root $VIDEO_ROOT \
        --variant b_topk_nofusion \
        --num-samples 150 --seed 99 \
        --train-seed 42 --train-max-samples 2000 \
        --out-dir runs/heldout_eval \
        --every 10"

# 1. Untrained vanilla Qwen3-Omni (no LoRA, no checkpoint)
#    This is the true prior-driven ceiling; should match E1 identity ~65%.
echo "=== [1/5] untrained ==="
python -u -m av_ib.eval.eval_v6_heldout $COMMON --label untrained

# 2. LoRA baseline (no VIB, beta=0) — did LoRA fine-tuning improve anything?
echo "=== [2/5] baseline (LoRA, no VIB) ==="
python -u -m av_ib.eval.eval_v6_heldout $COMMON \
    --label baseline \
    --ckpt-path runs/pilot_baseline/final.pt

# 3. Gate 3 pilot A: beta=1e-6 (barely any KL pressure)
echo "=== [3/5] beta=1e-6 ==="
python -u -m av_ib.eval.eval_v6_heldout $COMMON \
    --label gate3_beta1e6 \
    --ckpt-path runs/gate3_beta1e6/final.pt

# 4. Gate 3 pilot B: beta=7e-6 (sweet-spot candidate, 83% kl_v reduction)
echo "=== [4/5] beta=7e-6 ==="
python -u -m av_ib.eval.eval_v6_heldout $COMMON \
    --label gate3_beta7e6 \
    --ckpt-path runs/gate3_beta7e6/final.pt

# 5. Gate 3 pilot C: beta=3.5e-5 (aggressive, 94% kl_v reduction)
echo "=== [5/5] beta=3.5e-5 ==="
python -u -m av_ib.eval.eval_v6_heldout $COMMON \
    --label gate3_beta3e5 \
    --ckpt-path runs/gate3_beta3e5/final.pt

echo "=== Done $(date) ==="
