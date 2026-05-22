#!/bin/bash
#PBS -N eval_avh_transfer
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=01:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o eval_avhbench_transfer.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

VARIANT="${VARIANT:?VARIANT required}"
CKPT_DIR="${CKPT_DIR:?CKPT_DIR required, e.g. runs/musicavqa_v1_3ep}"
OUT_NAME="${OUT_NAME:?OUT_NAME required, e.g. musicavqa_v1_3ep}"

LOG_FILE="eval_avh_${OUT_NAME}.qsub.log"
echo "=== variant=$VARIANT ckpt=$CKPT_DIR out=$OUT_NAME ===" | tee "$LOG_FILE"

python -u scripts/eval_avhbench.py \
    --variant "$VARIANT" \
    --ckpt-dir "$CKPT_DIR" \
    --out-name "$OUT_NAME" 2>&1 | tee -a "$LOG_FILE"

echo "=== Done: $(date) ===" | tee -a "$LOG_FILE"
