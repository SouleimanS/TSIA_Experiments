#!/bin/bash
#PBS -N explain_e1_avqa
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o explain_e1_avqa.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/explain_e1_avqa_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

# ── ADJUST THESE to where the AVQA annotations + VGGSound clips live ──
ANN_PATH="$HOME/SOULEIMAN_repo/datasets/AVQA/train_qa.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/AVQA/videos"

if [ ! -f "$ANN_PATH" ]; then
    echo "ERROR: $ANN_PATH not found — download AVQA annotations first." >&2
    exit 1
fi

# GATE for the dataset switch: same per-modality ablation as on MUSIC-AVQA.
# If dropping audio costs real NLL here (unlike MUSIC-AVQA's +8.3%), the
# benchmark can train/test an AV bottleneck and the abstention pipeline moves.
echo "=== E1: AV-reliance ablation on AVQA (untrained model) ==="
python -u -m av_ib.eval.explain e1 \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --dataset avqa \
    --variant b_topk_nofusion \
    --num-samples 40 \
    --seed 42 \
    --every 5

echo "=== Done $(date) ==="
