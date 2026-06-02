#!/bin/bash
#PBS -N compare_musicavqa
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o compare_musicavqa.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

echo "=== Music-AVQA comparison table ==="
python -u av_ib/eval/eval_musicavqa.py \
    --compare \
        "fullb@1k:results/eval/musicavqa/fullb@1k/metrics.json" \
        "fullb:results/eval/musicavqa/fullb/metrics.json" \
        "musicavqa_1k:results/eval/musicavqa/musicavqa_1k/metrics.json"

echo "=== Done: $(date) ==="
