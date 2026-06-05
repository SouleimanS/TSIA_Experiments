#!/bin/bash
#PBS -N pilot_b_topk_nofusion_nf
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o pilot_b_topk_nofusion_noisefree.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"

mkdir -p runs/pilot_b_topk_nofusion_noisefree

# TIER 1 of the gate test: noise OFF (z=mu), beta=0. Isolates the splice +
# bottleneck path from the injected reparam noise. If this matches baseline but
# the noise-on pilot does not, the 0.22-std reparam noise is the culprit.
echo "=== PILOT (noise-free): v6 variant=b_topk_nofusion | 2k samples seed=42 ==="
python -u -m av_ib.train.train_v6 \
    --dataset music_avqa \
    --ann-path "$ANN_PATH" \
    --video-root "$VIDEO_ROOT" \
    --variant b_topk_nofusion \
    --max-samples 2000 --seed 42 \
    --num-steps 2000 \
    --lr 1e-4 \
    --beta-v 0 --beta-a 0 --beta-j 0 \
    --aux-weight 0.1 \
    --no-sample-noise \
    --log-path runs/pilot_b_topk_nofusion_noisefree/log.jsonl \
    --ckpt-path runs/pilot_b_topk_nofusion_noisefree/final.pt \
    --print-every 50

echo "=== Done $(date) ==="
