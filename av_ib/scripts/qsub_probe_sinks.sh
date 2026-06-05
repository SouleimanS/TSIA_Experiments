#!/bin/bash
#PBS -N probe_sinks
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o probe_sinks.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

# Write output to a fixed path so it's readable from any login node,
# independent of PBS log routing between login1/login2.
OUTFILE="$PBS_O_WORKDIR/runs/probe_sinks_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1

echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

echo "=== Sink probe: do {985,1992} fire on vision-encoder output? ==="
python -u -m av_ib.eval.probe_sinks \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_fusion \
    --num-samples 100

echo "=== Done $(date) ==="
