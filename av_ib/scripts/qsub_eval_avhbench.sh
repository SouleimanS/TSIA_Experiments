#!/bin/bash
#PBS -N eval_avhbench
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o eval_avhbench.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/eval_avhbench_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/avhbench_eval"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

QA_JSON="$HOME/SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0/json/qa.json"
VIDEO_DIR="$HOME/SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0/video"
VARIANT="b_topk_nofusion"
MAX=100   # cap at 100 items per run; full AVHBench can be run later

# Overfitting check: if trained models score far below vanilla on AVHBench
# (a dataset they never trained on), catastrophic forgetting has occurred.

# 1. Vanilla Qwen3-Omni — the ceiling reference (no fine-tuning)
echo "=== [1/5] vanilla Qwen3-Omni baseline ==="
python -u -m av_ib.eval.eval_avhbench_v6 \
    --baseline \
    --qa-json  "$QA_JSON" \
    --video-dir "$VIDEO_DIR" \
    --max-items $MAX \
    --out-csv  runs/avhbench_eval/vanilla.csv \
    --out-json runs/avhbench_eval/vanilla.json

# 2. LoRA baseline (no VIB, beta=0) — did LoRA alone cause forgetting?
echo "=== [2/5] LoRA baseline (no VIB) ==="
python -u -m av_ib.eval.eval_avhbench_v6 \
    --variant  "$VARIANT" \
    --ckpt-path runs/pilot_baseline/final.pt \
    --qa-json  "$QA_JSON" \
    --video-dir "$VIDEO_DIR" \
    --max-items $MAX \
    --out-csv  runs/avhbench_eval/lora_baseline.csv \
    --out-json runs/avhbench_eval/lora_baseline.json

# 3. Gate 3 beta=1e-6 (minimal KL pressure, mostly memorized)
echo "=== [3/5] gate3 beta=1e-6 ==="
python -u -m av_ib.eval.eval_avhbench_v6 \
    --variant  "$VARIANT" \
    --ckpt-path runs/gate3_beta1e6/final.pt \
    --qa-json  "$QA_JSON" \
    --video-dir "$VIDEO_DIR" \
    --max-items $MAX \
    --out-csv  runs/avhbench_eval/gate3_beta1e6.csv \
    --out-json runs/avhbench_eval/gate3_beta1e6.json

# 4. Gate 3 beta=7e-6 (sweet-spot candidate, 85% kl_v reduction)
echo "=== [4/5] gate3 beta=7e-6 ==="
python -u -m av_ib.eval.eval_avhbench_v6 \
    --variant  "$VARIANT" \
    --ckpt-path runs/gate3_beta7e6/final.pt \
    --qa-json  "$QA_JSON" \
    --video-dir "$VIDEO_DIR" \
    --max-items $MAX \
    --out-csv  runs/avhbench_eval/gate3_beta7e6.csv \
    --out-json runs/avhbench_eval/gate3_beta7e6.json

# 5. Gate 3 beta=3.5e-5 (aggressive compression)
echo "=== [5/5] gate3 beta=3.5e-5 ==="
python -u -m av_ib.eval.eval_avhbench_v6 \
    --variant  "$VARIANT" \
    --ckpt-path runs/gate3_beta3e5/final.pt \
    --qa-json  "$QA_JSON" \
    --video-dir "$VIDEO_DIR" \
    --max-items $MAX \
    --out-csv  runs/avhbench_eval/gate3_beta3e5.csv \
    --out-json runs/avhbench_eval/gate3_beta3e5.json

# Summary comparison
echo ""
echo "=== Summary comparison ==="
python -u - <<'EOF'
import json, glob, pathlib

files = sorted(pathlib.Path("runs/avhbench_eval").glob("*.json"))
print(f"\n{'Label':<22} {'Acc':>7} {'F1':>7}  Per-task")
print("-" * 80)
for fp in files:
    d = json.loads(fp.read_text())
    ov = d.get("overall", {})
    acc = ov.get("acc", float("nan"))
    f1  = ov.get("f1",  float("nan"))
    per = "  ".join(
        f"{t.split()[-1][:8]}={m['acc']*100:.0f}%"
        for t, m in d.get("per_task", {}).items()
    )
    print(f"{fp.stem:<22} {acc*100:>6.1f}%  {f1:>6.3f}  {per}")
EOF

echo "=== Done $(date) ==="
