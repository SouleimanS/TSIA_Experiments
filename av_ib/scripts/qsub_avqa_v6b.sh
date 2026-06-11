#!/bin/bash
#PBS -N avqa_v6b
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o avqa_v6b.qsub.log

# Condition 4: Version-6b architecture (video SinkAwareVIB + audio
# NormTopKSinkVIB, direct concat, frozen LLM) trained on AVQA with the
# rate penalty active on video (beta chosen from the MUSIC-AVQA sweep).

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/avqa_v6b_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/avqa_v6b"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/AVQA/train_qa.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/AVQA/videos"
[ -f "$ANN_PATH" ] || { echo "ERROR: $ANN_PATH missing — run download_avqa.py" >&2; exit 1; }

python -u -m av_ib.train.train_v6 \
    --dataset avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --num-steps 2000 \
    --lr 1e-4 \
    --beta-v 7e-6 --beta-a 7e-6 --beta-j 0 \
    --aux-weight 0.1 \
    --log-path runs/avqa_v6b/log.jsonl \
    --ckpt-path runs/avqa_v6b/final.pt \
    --save-every 500 \
    --print-every 50

echo "=== Done $(date) ==="
