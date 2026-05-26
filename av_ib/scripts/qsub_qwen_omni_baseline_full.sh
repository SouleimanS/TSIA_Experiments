#!/bin/bash
#PBS -N qwen_omni_full
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o qwen_omni_baseline_full.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-test.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

echo "=== Full val (9129 records, ~3h) ==="
python -u -m av_ib.eval.qwen_omni_baseline \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --output-csv qwen_omni_full.csv \
    --output-json qwen_omni_full.json

echo "=== Done $(date) ==="
