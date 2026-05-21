#!/bin/bash
# Submit all 12 noisy MUSIC-AVQA training jobs.
# Usage:    bash scripts/launch_noisy_sweep.sh
# Dry-run:  DRY_RUN=1 bash scripts/launch_noisy_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DRY_RUN="${DRY_RUN:-0}"

JOBS=(
    "v1 gaussian  0.1 v1_noisy_gauss_s0.1"
    "v3 gaussian  0.1 v3_noisy_gauss_s0.1"
    "v4 gaussian  0.1 v4_noisy_gauss_s0.1"
    "v1 gaussian  0.5 v1_noisy_gauss_s0.5"
    "v3 gaussian  0.5 v3_noisy_gauss_s0.5"
    "v4 gaussian  0.5 v4_noisy_gauss_s0.5"
    "v1 audio_mix 0.1 v1_noisy_mix_s0.1"
    "v3 audio_mix 0.1 v3_noisy_mix_s0.1"
    "v4 audio_mix 0.1 v4_noisy_mix_s0.1"
    "v1 audio_mix 0.5 v1_noisy_mix_s0.5"
    "v3 audio_mix 0.5 v3_noisy_mix_s0.5"
    "v4 audio_mix 0.5 v4_noisy_mix_s0.5"
)

echo "=== Noisy sweep: ${#JOBS[@]} jobs ==="
for job in "${JOBS[@]}"; do
    read -r variant mode sigma out <<< "$job"
    VARS="VARIANT=$variant,NOISE_MODE=$mode,NOISE_SIGMA=$sigma,OUT_NAME=$out"
    CMD="qsub -v $VARS scripts/qsub_train_musicavqa_noisy.sh"
    echo "  $CMD"
    if [ "$DRY_RUN" != "1" ]; then
        JOB_ID=$(eval "$CMD")
        echo "    -> $JOB_ID"
    fi
done
echo "=== Submitted. Check with: qstat -u \$USER ==="
