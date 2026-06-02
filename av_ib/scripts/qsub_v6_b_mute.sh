#!/bin/bash
#PBS -N v6b_mute
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o v6_b_mute.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

VIDEO_ROOT="/home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/corrupted_v6_1k/mute"

python -u -m av_ib.eval.v6_eval \
    --ckpt-path "runs/v6_tier1_b/step_1000.pt" \
    --ann-path "/home/aab11336im/SOULEIMAN_repo/TSIA_Experiments/av_ib/runs/v6_tier1_b/eval_subset_1000.json" \
    --video-root "$VIDEO_ROOT" \
    --num-records 1000 --seed 42 --variant b \
    --out-csv "runs/v6_tier1_b/eval_corrupted_mute_1000.csv"
echo "=== Done $(date) ==="
