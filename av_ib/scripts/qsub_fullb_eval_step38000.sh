#!/bin/bash
#PBS -N fullb_eval_38k
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o fullb_eval_step38000.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -u -m av_ib.eval.v6_eval \
    --ckpt-path "runs/v6_full_b/step_38000.pt" \
    --ann-path "runs/v6_tier1_b/eval_subset_1000.json" \
    --video-root "$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all" \
    --num-records 1000 --seed 42 --variant b \
    --out-csv "runs/v6_full_b/eval_step38000_1000.csv"
echo "Done $(date)"
