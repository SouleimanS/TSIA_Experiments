#!/bin/bash
# Submit 16 AVHBench transfer eval jobs (one per MUSIC-AVQA best.pt).
# Each takes ~10-15 min; runs in parallel up to ABCI slot limits.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"

# variant   run_dir_name (without runs/ prefix)
JOBS=(
    "v1     musicavqa_v1_3ep"
    "v1     musicavqa_v1_noisy_gauss_s0.1"
    "v1     musicavqa_v1_noisy_gauss_s0.5"
    "v1     musicavqa_v1_noisy_mix_s0.1"
    "v1     musicavqa_v1_noisy_mix_s0.5"
    "v2_nb1 musicavqa_v2_3ep_nb1"
    "v3     musicavqa_v3_3ep_b1e-2_prepatch"
    "v3     musicavqa_v3_noisy_gauss_s0.1"
    "v3     musicavqa_v3_noisy_gauss_s0.5"
    "v3     musicavqa_v3_noisy_mix_s0.1"
    "v3     musicavqa_v3_noisy_mix_s0.5"
    "v4     musicavqa_v4_3ep_b1e-2"
    "v4     musicavqa_v4_noisy_gauss_s0.1"
    "v4     musicavqa_v4_noisy_gauss_s0.5"
    "v4     musicavqa_v4_noisy_mix_s0.1"
    "v4     musicavqa_v4_noisy_mix_s0.5"
)

echo "=== AVHBench transfer eval: ${#JOBS[@]} jobs ==="
for job in "${JOBS[@]}"; do
    read -r variant run <<< "$job"
    CKPT="runs/$run"
    OUT="avh_transfer_$run"
    if [ ! -f "$CKPT/best.pt" ]; then
        echo "SKIP $run (no best.pt)"
        continue
    fi
    VARS="VARIANT=$variant,CKPT_DIR=$CKPT,OUT_NAME=$OUT"
    CMD="qsub -v $VARS scripts/qsub_eval_avhbench_transfer.sh"
    echo "  $CMD"
    if [ "$DRY_RUN" != "1" ]; then
        JOB_ID=$(eval "$CMD")
        echo "    -> $JOB_ID"
    fi
done
echo "=== Done ==="
