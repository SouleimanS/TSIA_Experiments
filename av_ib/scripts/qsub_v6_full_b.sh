#!/bin/bash
#PBS -N v6_full_b
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=96:00:00
#PBS -j oe
#PBS -o v6_full_b.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/qsub_v6_full_b_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/v6_full_b

echo "=== v6 full variant b: 2 epochs = 63854 steps, beta=0, lr=1e-4 ==="
python -u -m av_ib.train.train_v6 \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --num-steps 63854 \
    --lr 1e-4 \
    --beta-v 0 --beta-a 0 --beta-j 0 \
    --aux-weight 0.1 \
    --log-path runs/v6_full_b/log.jsonl \
    --ckpt-path runs/v6_full_b/final.pt \
    --save-every 2000 \
    --print-every 100 \
    --variant b

echo "=== Done $(date) ==="
