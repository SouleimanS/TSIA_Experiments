#!/bin/bash
#PBS -N t1_aux0
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o t1_aux0.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/t1_aux0_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/t1_aux0

# T1: identical to gate3_beta7e6 EXCEPT aux_weight=0.
# Tests confound #1 (aux-head dominance): the gate3 loss was 85% aux head.
# With aux removed, the VIB is driven ONLY by NLL + beta*KL — the actual IB
# objective. If held-out improves vs gate3_beta7e6, the aux heads were the
# problem. One-variable change for a clean attribution.
echo "=== T1: LoRA + aux_weight=0 | beta_v=beta_a=7e-6 | 2k samples seed=42 ==="
python -u -m av_ib.train.train_v6 \
    --dataset music_avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --max-samples 2000 --seed 42 \
    --num-steps 2000 \
    --lr 1e-4 \
    --beta-v 7e-6 --beta-a 7e-6 --beta-j 0 \
    --aux-weight 0 \
    --log-path runs/t1_aux0/log.jsonl \
    --ckpt-path runs/t1_aux0/final.pt \
    --print-every 50

echo "=== Done $(date) ==="
