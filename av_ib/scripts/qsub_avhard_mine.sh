#!/bin/bash
#PBS -N avhard_mine
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o avhard_mine.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/avhard_mine_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0   # single GPU: avoid cross-device splice errors

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

# Mine the AV-essential subset on the UNTRAINED vanilla model (confound-free).
# Behavioral definition: id correct, audio_only WRONG, video_only WRONG, not
# zero-solvable -> the answer needs BOTH modalities. Scans 400 candidates.
echo "=== AV-HARD mining | untrained model | 400 candidates ==="
python -u -m av_ib.eval.explain avhard \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --num-samples 400 --seed 7 --every 20 \
    --out-json runs/avhard_subset.json

echo "=== Done $(date) ==="
