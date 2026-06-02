#!/bin/bash
#PBS -N dsink_audio_validate
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o dsink_audio_validate.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

export FORCE_QWENVL_VIDEO_READER=decord

echo "=== Validating audio sink dims [1992, 940, 1312, 985] ==="
python -u -m av_ib.eval.validate_dsink_proxy \
    --ann-path    /home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-test.json \
    --video-root  /home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all \
    --dsink-dims  1992 940 1312 985 \
    --num-records 50 \
    --tau         5.0 \
    --top-k-frac  0.05 \
    --out-dir     results/validate_dsink_audio

echo "=== Done: $(date) ==="
