#!/bin/bash
#PBS -N smoke_avqa
#PBS -l walltime=01:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o smoke_avqa.qsub.log

# ----- Resource selection -----
# On ABCI 3.0 the resource type is set via -l rt_HF=1 (1 H100 share),
# rt_HG=1, rt_HC=1, etc. Adjust if your group uses a different RT.
# (kept in script body comment; PBS directive form below is the standard)

set -euo pipefail

cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname) ==="
echo "=== Date: $(date) ==="
echo "=== GPUs: ==="
nvidia-smi -L || true
echo ""

# Activate conda env
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

echo "=== Python / CUDA ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0))"
echo ""

echo "=== Running smoke_train_avqa.py ==="
python scripts/smoke_train_avqa.py
