#!/bin/bash
#PBS -N eval_avh_para
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -j oe
#PBS -o eval_avhbench_paraphrase.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

VARIANT="${VARIANT:-v3}"
CKPT_DIR="${CKPT_DIR:-runs/avqa_v3_3ep_b1e-2}"
OUT_NAME="${OUT_NAME:-v3_b1e-2_test}"
TEST_FILE="${TEST_FILE:-data/avhbench_split_test.json}"

echo "=== Paraphrase eval ==="
echo "  variant=$VARIANT  ckpt=$CKPT_DIR  out=$OUT_NAME  test=$TEST_FILE"
python -u scripts/eval_avhbench_paraphrase.py \
    --variant "$VARIANT" \
    --ckpt-dir "$CKPT_DIR" \
    --out-name "$OUT_NAME" \
    --test-file "$TEST_FILE"

echo "=== Done: $(date) ==="
