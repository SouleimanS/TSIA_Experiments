#!/bin/bash
#PBS -N avhbench_fullb
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o avhbench_fullb.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== $(hostname) $(date) ==="
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
CKPT="runs/v6_full_b/step_38000.pt"
[ -f "$CKPT" ] || CKPT="runs/v6_full_b/final.pt"
python -u -m av_ib.eval.v6_avhbench_eval \
    --ckpt-path "$CKPT" \
    --variant b \
    --out-csv  results/avhbench_fullb_step38000.csv \
    --out-json results/avhbench_fullb_step38000.json
echo "=== Done $(date) ==="
