#!/bin/bash
#PBS -N extract_dsink_im
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o extract_dsink_imstart.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"
SAMPLE_VIDEO="$VIDEO_ROOT/00000093.mp4"

python -u -m av_ib.eval.extract_dsink \
    --bos-id 151644 \
    --record-video-path "$SAMPLE_VIDEO" \
    --record-prompt "How many musical instruments were heard?" \
    --top-k 4 \
    --out-dir results/dsink_imstart/

echo "=== Done $(date) ==="
