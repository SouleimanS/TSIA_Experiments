#!/bin/bash
#PBS -N gate3_beta1e6
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o gate3_beta1e6.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/gate3_beta1e6_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/gate3_beta1e6

# GATE 3 pilot A: beta=1e-6 (beta*kl_v ~ 1.4% of NLL at init).
# Minimal KL pressure — expect kl_mu_term to start shrinking only slightly.
# This is the "safe" end of the sweep; NLL should stay near the beta=0 baseline.
echo "=== GATE 3 pilot: beta_v=1e-6 | 2k samples seed=42 ==="
python -u -m av_ib.train.train_v6 \
    --dataset music_avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --max-samples 2000 --seed 42 \
    --num-steps 2000 \
    --lr 1e-4 \
    --beta-v 1e-6 --beta-a 1e-6 --beta-j 0 \
    --aux-weight 0.1 \
    --log-path runs/gate3_beta1e6/log.jsonl \
    --ckpt-path runs/gate3_beta1e6/final.pt \
    --print-every 50

echo "=== Done $(date) ==="
