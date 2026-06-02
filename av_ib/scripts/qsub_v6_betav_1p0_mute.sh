#!/bin/bash
#PBS -N v6_1p0_mute
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o v6_betav_1p0_mute.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u -m av_ib.eval.v6_eval \
    --ckpt-path "runs/v6_betav_1p0/step_1000.pt" \
    --ann-path "/home/aab11336im/SOULEIMAN_repo/TSIA_Experiments/av_ib/runs/v6_tier1_b/eval_subset_1000.json" \
    --video-root "/home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/corrupted_v6_1k/mute" \
    --num-records 1000 --seed 42 --variant b \
    --out-csv "runs/v6_betav_1p0/eval_corrupted_mute_1000.csv"
echo "=== Done $(date) ==="
