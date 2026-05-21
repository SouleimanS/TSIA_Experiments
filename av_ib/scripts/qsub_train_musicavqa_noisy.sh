#!/bin/bash
#PBS -N train_mavqa_noisy
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=14:00:00
#PBS -l select=1
#PBS -j oe
#PBS -o train_musicavqa_noisy.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

VARIANT="${VARIANT:?VARIANT required}"
NOISE_MODE="${NOISE_MODE:?NOISE_MODE required}"
NOISE_SIGMA="${NOISE_SIGMA:?NOISE_SIGMA required}"
OUT_NAME="${OUT_NAME:?OUT_NAME required}"
OUT_DIR="runs/musicavqa_${OUT_NAME}"
LOG_FILE="train_musicavqa_${OUT_NAME}.qsub.log"

echo "=== variant=$VARIANT mode=$NOISE_MODE sigma=$NOISE_SIGMA out=$OUT_DIR ===" | tee "$LOG_FILE"

case "$VARIANT" in
    v1)
        python -u scripts/train_musicavqa_v1_noisy.py \
            --noise-mode "$NOISE_MODE" --noise-sigma "$NOISE_SIGMA" \
            --output-dir "$OUT_DIR" 2>&1 | tee -a "$LOG_FILE"
        ;;
    v3)
        BETA="${BETA:-1e-2}"
        python -u scripts/train_musicavqa_v3_noisy.py \
            --beta "$BETA" \
            --noise-mode "$NOISE_MODE" --noise-sigma "$NOISE_SIGMA" \
            --output-dir "$OUT_DIR" 2>&1 | tee -a "$LOG_FILE"
        ;;
    v4)
        BETA_V="${BETA_V:-1e-2}"; BETA_A="${BETA_A:-1e-2}"; BETA_J="${BETA_J:-1e-2}"; AUX_WEIGHT="${AUX_WEIGHT:-1.0}"
        python -u scripts/train_musicavqa_v4_noisy.py \
            --beta-v "$BETA_V" --beta-a "$BETA_A" --beta-j "$BETA_J" --aux-weight "$AUX_WEIGHT" \
            --noise-mode "$NOISE_MODE" --noise-sigma "$NOISE_SIGMA" \
            --output-dir "$OUT_DIR" 2>&1 | tee -a "$LOG_FILE"
        ;;
    *)
        echo "ERROR: unknown VARIANT=$VARIANT" | tee -a "$LOG_FILE"
        exit 1
        ;;
esac
echo "=== Done: $(date) ===" | tee -a "$LOG_FILE"
