#!/bin/bash
#PBS -N base_shift
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o baseline_shift.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

VIDEO_ROOT="/home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/corrupted_v6_1k/shift"
mkdir -p "runs/corrupted_baseline_shift"

python -u -m av_ib.eval.qwen_omni_baseline \
    --ann-path "/home/aab11336im/SOULEIMAN_repo/TSIA_Experiments/av_ib/runs/v6_tier1_b/eval_subset_1000.json" \
    --video-root "$VIDEO_ROOT" \
    --max-records 1000 \
    --output-csv "runs/corrupted_baseline_shift/eval_1000.csv" \
    --output-json "runs/corrupted_baseline_shift/eval_1000.json"
echo "=== Done $(date) ==="
