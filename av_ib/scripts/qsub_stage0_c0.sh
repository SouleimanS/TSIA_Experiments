#!/bin/bash
#PBS -N stage0_c0
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o stage0_c0.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/stage0_c0_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/stage0"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"
QA_JSON="$HOME/SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0/json/qa.json"
VIDEO_DIR="$HOME/SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0/video"

# STAGE 0 — zero-training baseline. No new training needed.
# Tests whether the abstain PROMPT alone (C0) or T2+prompt already fail-safe
# on AVHBench Video-driven Audio Hallucination (the task T2 regressed −8.3pp).
#
# Uses eval_riskcoverage.py with identity corruption only (no training yet)
# to measure current abstain rate under the new abstain-prompt.
# Also re-runs vanilla vs T2 on AVHBench with the abstain prompt added,
# so we can see if T2+prompt already moves coverage on the hallucination task.

echo "=== [1/3] C0: vanilla + abstain prompt | risk-coverage on MUSIC-AVQA held-out ==="
python -u -m av_ib.eval.eval_riskcoverage \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_video_only \
    --num-samples 150 --seed 99 --every 15 \
    --out-json runs/stage0/c0_musicavqa.json

echo "=== [2/3] C0+T2: T2 checkpoint + abstain prompt | risk-coverage on MUSIC-AVQA held-out ==="
python -u -m av_ib.eval.eval_riskcoverage \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_video_only \
    --ckpt-path runs/t2_nolora_b7e6/final.pt \
    --num-samples 150 --seed 99 --every 15 \
    --out-json runs/stage0/c0_t2_musicavqa.json

echo "=== [3/3] summary ==="
python -u - <<'EOF'
import json, pathlib
for fp in sorted(pathlib.Path("runs/stage0").glob("*.json")):
    d = json.loads(fp.read_text())
    s = d["summary"]
    print(f"\n{'='*60}")
    print(f"  {fp.stem}")
    print(f"  {'corruption':<14} {'coverage':>9} {'risk':>8} {'abstained':>10}")
    for corr, a in s.items():
        print(f"  {corr:<14} {a['coverage']:>8.1%} {a['risk']:>8.1%} {a['n_abstained']:>10}/{a['n']}")
EOF

echo "=== Done $(date) ==="
