#!/bin/bash
#PBS -N smoke_noisy
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -j oe
#PBS -o smoke_noisy.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

echo ""
echo "=== Smoke 1: v1 + gaussian @ 0.1 (20 steps) ==="
python -u scripts/train_musicavqa_v1_noisy.py \
    --noise-mode gaussian --noise-sigma 0.1 \
    --output-dir /tmp/smoke_v1_gauss \
    --smoke-steps 20

echo ""
echo "=== Smoke 2: v3 + gaussian @ 0.5 (20 steps) ==="
python -u scripts/train_musicavqa_v3_noisy.py \
    --beta 1e-2 \
    --noise-mode gaussian --noise-sigma 0.5 \
    --output-dir /tmp/smoke_v3_gauss \
    --smoke-steps 20

echo ""
echo "=== Smoke 3: v4 + audio_mix @ 0.5 (20 steps) — riskiest path ==="
python -u scripts/train_musicavqa_v4_noisy.py \
    --beta-v 1e-2 --beta-a 1e-2 --beta-j 1e-2 --aux-weight 1.0 \
    --noise-mode audio_mix --noise-sigma 0.5 \
    --output-dir /tmp/smoke_v4_mix \
    --smoke-steps 20

echo ""
echo "=== ALL SMOKES PASSED: $(date) ==="
