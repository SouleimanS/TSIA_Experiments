#!/bin/bash
#PBS -N val_dsink_smoke
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o validate_dsink_smoke.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-test.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

python -u -m av_ib.eval.validate_dsink_proxy \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --dsink-dims 985 1992 \
    --num-records 5 \
    --tau 5.0 \
    --top-k-frac 0.05 \
    --out-dir results/validate_dsink_smoke/

echo "=== Done $(date) ==="
