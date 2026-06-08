#!/bin/bash
#PBS -N explain_e1
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o explain_e1.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/explain_e1_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

# E1 on the UNTRAINED model = vanilla Qwen3-Omni's AV reliance (no checkpoint,
# no confounds). THE gate: if mean-replacing AV tokens barely moves NLL, the
# model is prior-driven and the whole IB program needs a more AV-dependent task.
echo "=== E1: AV-reliance ablation (untrained model) ==="
python -u -m av_ib.eval.explain e1 \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --num-samples 40 \
    --seed 42 \
    --every 5

echo "=== Done $(date) ==="
