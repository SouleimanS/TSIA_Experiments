#!/bin/bash
#PBS -N v5_train_smoke
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o v5_train_smoke.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-test.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

echo "=== v5 forward_train smoke ==="
python -u -m av_ib.eval.v5_train_smoke \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT"

echo "=== Done $(date) ==="
