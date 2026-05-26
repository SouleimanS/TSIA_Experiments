#!/bin/bash
#PBS -N inspect_qwen_omni
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o inspect_qwen_omni.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
df -h ~/.cache 2>/dev/null | tail -2

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

echo "=== transformers version ==="
python -c "import transformers; print(transformers.__version__)"

echo "=== Starting Qwen3-Omni inspection (will download ~60GB on first run) ==="
python -u scripts/inspect_qwen_omni.py
echo "=== Done $(date) ==="
