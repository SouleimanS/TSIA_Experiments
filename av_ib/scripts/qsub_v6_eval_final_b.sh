#!/bin/bash
#PBS -N v6_eval_final_b
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o v6_eval_final_b.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-test.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"
for STEP in 1000; do
    CKPT="runs/v6_tier1_b/step_${STEP}.pt"
    OUT="runs/v6_tier1_b/eval_step${STEP}_1000.csv"
    if [ ! -f "$CKPT" ]; then echo "Skip (no ckpt): $CKPT"; continue; fi
    if [ -f "$OUT" ]; then echo "Skip (done): $OUT"; continue; fi
    echo "=== variant b, step $STEP ==="
    python -u -m av_ib.eval.v6_eval \
        --ckpt-path "$CKPT" --ann-path "$ANN_PATH" --video-root "$VIDEO_ROOT" \
        --num-records 1000 --seed 42 --variant b --out-csv "$OUT"
done
echo "=== Done $(date) ==="
