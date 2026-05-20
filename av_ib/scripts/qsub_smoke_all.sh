#!/bin/bash
#PBS -N smoke_all
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -j oe
#PBS -o smoke_all.qsub.log

set -uo pipefail   # no -e: we want to continue past per-config failures
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

SMOKE_STEPS=3
SMOKE_BASE=runs/_smoke

# Wipe any previous smoke artifacts so they don't confuse the log
rm -rf "$SMOKE_BASE"
mkdir -p "$SMOKE_BASE"

run_smoke() {
    local name="$1"
    local cmd="$2"
    echo ""
    echo "============================================================"
    echo "=== SMOKE: $name"
    echo "=== CMD:   $cmd"
    echo "============================================================"
    eval "$cmd"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "*** $name: PASS ***"
    else
        echo "*** $name: FAIL (exit $rc) ***"
    fi
}

# v2 variants
run_smoke "v2_nb1" "python -u scripts/train_avqa_v2_3ep.py --n-blocks 1 --output-dir $SMOKE_BASE/v2_nb1 --smoke-steps $SMOKE_STEPS"
run_smoke "v2_nb2" "python -u scripts/train_avqa_v2_3ep.py --n-blocks 2 --output-dir $SMOKE_BASE/v2_nb2 --smoke-steps $SMOKE_STEPS"

# v3 beta sweep
run_smoke "v3_b1e-4" "python -u scripts/train_avqa_v3_3ep.py --beta 1e-4 --output-dir $SMOKE_BASE/v3_b1e-4 --smoke-steps $SMOKE_STEPS"
run_smoke "v3_b1e-3" "python -u scripts/train_avqa_v3_3ep.py --beta 1e-3 --output-dir $SMOKE_BASE/v3_b1e-3 --smoke-steps $SMOKE_STEPS"
run_smoke "v3_b1e-2" "python -u scripts/train_avqa_v3_3ep.py --beta 1e-2 --output-dir $SMOKE_BASE/v3_b1e-2 --smoke-steps $SMOKE_STEPS"
run_smoke "v3_b1e-1" "python -u scripts/train_avqa_v3_3ep.py --beta 1e-1 --output-dir $SMOKE_BASE/v3_b1e-1 --smoke-steps $SMOKE_STEPS"
run_smoke "v3_b1e0"  "python -u scripts/train_avqa_v3_3ep.py --beta 1.0  --output-dir $SMOKE_BASE/v3_b1e0  --smoke-steps $SMOKE_STEPS"

echo ""
echo "=== All smoke runs done: $(date) ==="
echo ""
echo "=== Summary ==="
grep -E "SMOKE OK|FAIL" smoke_all.qsub.log || echo "(see above output)"
