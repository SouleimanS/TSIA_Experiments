#!/bin/bash
#PBS -N diag_drift
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o diag_drift.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "########## SINK-AWARE ##########"
python -u scripts/diag_splice_drift.py runs/v6_tier1_b/step_1000.pt b sink
echo ""
echo "########## STANDARD-VIB ##########"
python -u scripts/diag_splice_drift.py runs/v6_tier1_b_stdvib/step_1000.pt b standard
