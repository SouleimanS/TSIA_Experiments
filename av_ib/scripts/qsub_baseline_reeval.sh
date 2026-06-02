#!/bin/bash
#PBS -N baseline_reeval
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o baseline_reeval.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -u -m av_ib.eval.qwen_omni_baseline \
    --ann-path "runs/v6_tier1_b/eval_subset_1000.json" \
    --video-root "$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all" \
    --max-records 1000 \
    --output-csv "runs/baseline_reeval_1000.csv"
echo "Done $(date)"
