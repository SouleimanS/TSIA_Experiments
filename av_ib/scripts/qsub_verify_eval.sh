#!/bin/bash
#PBS -N verify_eval
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -j oe
#PBS -o verify_eval.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
python -u scripts/verify_eval_function.py
