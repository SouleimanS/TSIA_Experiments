#!/bin/bash
#PBS -N inspect_labels
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o inspect_labels.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

python -u -m av_ib.eval.inspect_labels \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT"

echo "=== Done $(date) ==="
