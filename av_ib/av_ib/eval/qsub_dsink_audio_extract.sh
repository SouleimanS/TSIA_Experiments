#!/bin/bash
#PBS -N dsink_audio_extract
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o dsink_audio_extract.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

echo "=== Extracting audio sink dims ==="
python -u -m av_ib.eval.extract_dsink \
    --record-video-path /home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/synthetic/esa00000004.mp4 \
    --record-prompt "What instrument is being played?" \
    --top-k 8 \
    --out-dir results/dsink_audio

echo "=== Done: $(date) ==="
