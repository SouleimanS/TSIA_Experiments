#!/bin/bash
#PBS -N abstain_labels
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -o abstain_labels.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/abstain_labels_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0   # single GPU: VIB hooks need same-device tensors

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

# Long pole for the abstention experiment: label the train pool with empirical
# abstain targets. flip = model right under identity, wrong under {zero,mean}.
# 3000 candidates on seed=42 (the training seed) to gather enough abstainable
# items. ~3 generations each; budget 10h.
echo "=== abstain-label mining | untrained model | 3000 candidates | seed=42 ==="
python -u -m av_ib.eval.explain alabels \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_video_only \
    --num-samples 3000 --seed 42 --every 50 \
    --out-json runs/abstain_labels.json

echo "=== Done $(date) ==="
