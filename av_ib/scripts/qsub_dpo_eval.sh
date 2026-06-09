#!/bin/bash
#PBS -N dpo_eval
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o dpo_eval.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/dpo_eval_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/dpo_eval"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

# The decision test: risk-coverage on HELD-OUT corruptions (vid_noise, vid_scale)
# + negative control (aud_zero/mean). C2 >> C1 on held-out => bottleneck buys
# fail-safe generalization. Same 150-sample held-out set (seed=99) as Stage-0.

for cond in c1_dpo c2a_dpo c2b_dpo; do
    CKPT="runs/dpo/${cond}_final.pt"
    [ -f "$CKPT" ] || { echo "skip $cond: $CKPT missing"; continue; }
    LORA=""
    case "$cond" in
        c1_dpo|c2b_dpo) LORA="" ;;  # ckpt load sets use_lora via riskcoverage --ckpt-path
    esac
    echo "=== risk-coverage | $cond ==="
    python -u -m av_ib.eval.eval_riskcoverage \
        --ann-path "$ANN_PATH" --video-root "$VIDEO_ROOT" \
        --variant b_video_only --ckpt-path "$CKPT" \
        --num-samples 150 --seed 99 --every 15 \
        --out-json "runs/dpo_eval/${cond}.json"
done

echo "=== summary ==="
python -u - <<'EOF'
import json, pathlib
for fp in sorted(pathlib.Path("runs/dpo_eval").glob("*.json")):
    d = json.loads(fp.read_text()); s = d["summary"]
    print(f"\n{'='*60}\n  {fp.stem}")
    print(f"  {'corruption':<14}{'coverage':>10}{'risk':>9}")
    for corr, a in s.items():
        print(f"  {corr:<14}{a['coverage']:>9.1%}{a['risk']:>9.1%}")
EOF

echo "=== Done $(date) ==="
