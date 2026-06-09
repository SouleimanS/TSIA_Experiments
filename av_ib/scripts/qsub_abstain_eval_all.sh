#!/bin/bash
#PBS -N abstain_eval_all
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o abstain_eval_all.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/abstain_eval_all_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/abstain_eval"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

# 6 conditions: C1/C2a/C2b × SFT/DPO.
# Eval on the same held-out set (150 samples, seed=99) as Stage-0, so we can
# compare directly with the Stage-0 C0 baseline from runs/stage0/c0_musicavqa.json.
# Key test: held-out corruptions (vid_noise, vid_scale — never seen in training)
# tell us if generalization is real. Negative controls (aud_zero/mean) tell us
# if coverage is preserved when only the inert modality is degraded.

declare -A COND_CKPT=(
    [c1_sft]="runs/sft/c1_sft_final.pt"
    [c2a_sft]="runs/sft/c2a_sft_final.pt"
    [c2b_sft]="runs/sft/c2b_sft_final.pt"
    [c1_dpo]="runs/dpo/c1_dpo_final.pt"
    [c2a_dpo]="runs/dpo/c2a_dpo_final.pt"
    [c2b_dpo]="runs/dpo/c2b_dpo_final.pt"
)

for cond in c1_sft c2a_sft c2b_sft c1_dpo c2a_dpo c2b_dpo; do
    CKPT="${COND_CKPT[$cond]}"
    OUT="runs/abstain_eval/${cond}.json"
    [ -f "$CKPT" ] || { echo "SKIP $cond: $CKPT not found"; continue; }
    [ -f "$OUT"  ] && { echo "SKIP $cond: $OUT already exists"; continue; }

    echo "=== [$cond] ==="
    python -u -m av_ib.eval.eval_riskcoverage \
        --ann-path "$ANN_PATH" --video-root "$VIDEO_ROOT" \
        --variant b_video_only --ckpt-path "$CKPT" \
        --num-samples 150 --seed 99 --every 15 \
        --out-json "$OUT"
done

# ── comparison table ──
echo ""
echo "=== COMPARISON: SFT vs DPO (+ Stage-0 baseline) ==="
python -u - <<'PYEOF'
import json, pathlib

GROUPS = [
    ("TRAIN corrs",      ["identity", "vid_zero", "vid_mean"]),
    ("HELD-OUT (gen.)",  ["vid_noise", "vid_scale"]),
    ("NEG ctrl",         ["aud_zero",  "aud_mean"]),
]
CONDS = ["c1_sft", "c2a_sft", "c2b_sft", "c1_dpo", "c2a_dpo", "c2b_dpo"]

# Load Stage-0 baseline (C0: no training) for direct comparison
stage0 = {}
for fp in [pathlib.Path("runs/stage0/c0_musicavqa.json"),
           pathlib.Path("runs/stage0/c0_t2_musicavqa.json")]:
    if fp.exists():
        d = json.loads(fp.read_text())
        stage0[fp.stem] = d["summary"]

results = {}
for cond in CONDS:
    fp = pathlib.Path(f"runs/abstain_eval/{cond}.json")
    if fp.exists():
        results[cond] = json.loads(fp.read_text())["summary"]

for group_name, corrs in GROUPS:
    print(f"\n{'='*80}")
    print(f"  {group_name}")
    print(f"  {'corr':<12} {'cond':<12} {'coverage':>10} {'risk':>8}")
    for corr in corrs:
        # Stage-0 baselines first
        for sname, ss in stage0.items():
            if corr in ss:
                a = ss[corr]
                print(f"  {corr:<12} {('['+sname+']'):<12} {a['coverage']:>9.1%} {a['risk']:>8.1%}")
        # Trained conditions
        for cond, ss in results.items():
            if corr in ss:
                a = ss[corr]
                print(f"  {corr:<12} {cond:<12} {a['coverage']:>9.1%} {a['risk']:>8.1%}")
PYEOF

echo "=== Done $(date) ==="
