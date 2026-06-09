#!/bin/bash
#PBS -N avhb_t2
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o avhb_t2.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/avhb_t2_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs/avhbench_t2"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0   # single GPU: VIB hooks need same-device tensors

QA_JSON="$HOME/SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0/json/qa.json"
VIDEO_DIR="$HOME/SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0/video"
VARIANT="b_topk_nofusion"
MAX=200   # bigger sample than the earlier 100-item runs for a tighter estimate

# Best bottleneck config (frozen LLM + VIB only, beta=7e-6) vs vanilla.
# The no-lora ckpt loads its 16 VIB tensors into a use_lora=True eval model;
# LoRA stays zero-init (identity), so this runs as frozen-LLM + VIB as trained.

echo "=== [1/2] vanilla Qwen3-Omni baseline ==="
python -u -m av_ib.eval.eval_avhbench_v6 \
    --baseline \
    --qa-json  "$QA_JSON" \
    --video-dir "$VIDEO_DIR" \
    --max-items $MAX \
    --out-csv  runs/avhbench_t2/vanilla.csv \
    --out-json runs/avhbench_t2/vanilla.json

echo "=== [2/2] T2 (frozen LLM + VIB, beta=7e-6) ==="
python -u -m av_ib.eval.eval_avhbench_v6 \
    --variant  "$VARIANT" \
    --ckpt-path runs/t2_nolora_b7e6/final.pt \
    --qa-json  "$QA_JSON" \
    --video-dir "$VIDEO_DIR" \
    --max-items $MAX \
    --out-csv  runs/avhbench_t2/t2_b7e6.csv \
    --out-json runs/avhbench_t2/t2_b7e6.json

echo ""
echo "=== Summary comparison ==="
python -u - <<'EOF'
import json, pathlib
for fp in sorted(pathlib.Path("runs/avhbench_t2").glob("*.json")):
    d = json.loads(fp.read_text())
    ov = d.get("overall", {})
    per = "  ".join(
        f"{t.split()[-1][:8]}={m['acc']*100:.0f}%"
        for t, m in d.get("per_task", {}).items()
    )
    print(f"{fp.stem:<14} acc={ov.get('acc',float('nan'))*100:5.1f}%  "
          f"f1={ov.get('f1',float('nan')):.3f}   {per}")
EOF

echo "=== Done $(date) ==="
