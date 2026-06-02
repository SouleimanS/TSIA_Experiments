#!/bin/bash
#PBS -N avhbench_base
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o avhbench_baseline.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== $(hostname) $(date) ==="
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -u -m av_ib.eval.v6_avhbench_eval \
    --baseline \
    --out-csv  results/avhbench_baseline.csv \
    --out-json results/avhbench_baseline.json
echo "=== Done $(date) ==="
