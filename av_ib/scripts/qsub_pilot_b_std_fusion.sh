#!/bin/bash
#PBS -N pilot_b_std_fusion
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o pilot_b_std_fusion.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/qsub_pilot_b_std_fusion_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/pilot_b_std_fusion

echo "=== PILOT: v6 variant=b_std_fusion | 2k samples seed=42 ==="
python -u -m av_ib.train.train_v6 \
    --dataset music_avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_std_fusion \
    --max-samples 2000 --seed 42 \
    --num-steps 2000 \
    --lr 1e-4 \
    --beta-v 0 --beta-a 0 --beta-j 0 \
    --aux-weight 0.1 \
    --log-path runs/pilot_b_std_fusion/log.jsonl \
    --ckpt-path runs/pilot_b_std_fusion/final.pt \
    --print-every 50

echo "=== Done $(date) ==="
