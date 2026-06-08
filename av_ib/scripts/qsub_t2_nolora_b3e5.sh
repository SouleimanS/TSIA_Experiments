#!/bin/bash
#PBS -N t2_nolora_b3e5
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o t2_nolora_b3e5.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/t2_nolora_b3e5_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/t2_nolora_b3e5

# T2b: second point on the frozen-LLM rate-distortion curve (stronger beta).
# Together with T2 (beta=7e-6) and a beta=0 point this traces how much the
# FIXED model's NLL degrades as AV tokens are compressed — the core IB curve.
echo "=== T2b: frozen LLM + aux=0 | beta_v=beta_a=3.5e-5 | 2k samples seed=42 ==="
python -u -m av_ib.train.train_v6 \
    --dataset music_avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --no-lora \
    --max-samples 2000 --seed 42 \
    --num-steps 2000 \
    --lr 1e-4 \
    --beta-v 3.5e-5 --beta-a 3.5e-5 --beta-j 0 \
    --aux-weight 0 \
    --log-path runs/t2_nolora_b3e5/log.jsonl \
    --ckpt-path runs/t2_nolora_b3e5/final.pt \
    --print-every 50

echo "=== Done $(date) ==="
