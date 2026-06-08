#!/bin/bash
#PBS -N kl_study
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o kl_study.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/kl_study_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

# NLL reference: use the baseline pilot final NLL (~2.0 at step 0, lower after training).
# If you have a completed baseline pilot, paste its final step NLL here.
REFERENCE_NLL=2.0

echo "=== KL study: decompose mu-term vs var-term, beta sensitivity ==="
python -u -m av_ib.eval.kl_study \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --num-samples 50 \
    --reference-nll "$REFERENCE_NLL" \
    --every 10

echo "=== Done $(date) ==="
