#!/bin/bash
#PBS -N t6_e1_ckpts
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o t6_e1_ckpts.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/t6_e1_ckpts_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"
COMMON="--ann-path $ANN_PATH --video-root $VIDEO_ROOT \
        --variant b_topk_nofusion --num-samples 40 --seed 42 --every 10"

# T6: run the E1 AV-reliance ablation on the TRAINED checkpoints.
# THE question: does a trained bottleneck reduce the mean-replace damage
# (the hallucination direction)? Compare each checkpoint's `mean` row
# dNLL% and ans_acc against the untrained reference. Same seed=42 / 40
# samples as the original untrained E1, so the rows are directly comparable.

echo "##### T6 [1/4] untrained reference #####"
python -u -m av_ib.eval.explain e1 $COMMON

echo "##### T6 [2/4] gate3 beta=1e-6 #####"
python -u -m av_ib.eval.explain e1 $COMMON --ckpt-path runs/gate3_beta1e6/final.pt

echo "##### T6 [3/4] gate3 beta=7e-6 #####"
python -u -m av_ib.eval.explain e1 $COMMON --ckpt-path runs/gate3_beta7e6/final.pt

echo "##### T6 [4/4] gate3 beta=3.5e-5 #####"
python -u -m av_ib.eval.explain e1 $COMMON --ckpt-path runs/gate3_beta3e5/final.pt

echo "=== Done $(date) ==="
