#!/bin/bash
#PBS -N gate3_beta7e6
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o gate3_beta7e6.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/gate3_beta7e6_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/gate3_beta7e6

# GATE 3 pilot B: beta=7e-6 (beta*kl_v ~ 10% of NLL at init).
# Light compression — expect visible kl_mu_term reduction over 2k steps
# without NLL degradation. Sweet spot candidate for full training.
echo "=== GATE 3 pilot: beta_v=7e-6 | 2k samples seed=42 ==="
python -u -m av_ib.train.train_v6 \
    --dataset music_avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --max-samples 2000 --seed 42 \
    --num-steps 2000 \
    --lr 1e-4 \
    --beta-v 7e-6 --beta-a 7e-6 --beta-j 0 \
    --aux-weight 0.1 \
    --log-path runs/gate3_beta7e6/log.jsonl \
    --ckpt-path runs/gate3_beta7e6/final.pt \
    --print-every 50

echo "=== Done $(date) ==="
