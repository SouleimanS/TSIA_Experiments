#!/bin/bash
#PBS -N sinksym_smoke
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o sinksym_smoke.qsub.log
set -euo pipefail
cd "$PBS_O_WORKDIR"
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"
mkdir -p runs/sinksym_smoke
echo "=== SMOKE: variant b, video_vib=sink, fusion=sink_sym, 10 steps ==="
python -u -m av_ib.train.train_v6 \
    --ann-path "$ANN_PATH" --video-root "$VIDEO_ROOT" \
    --num-steps 10 --lr 1e-4 \
    --beta-v 0 --beta-a 0 --beta-j 0 --aux-weight 0.1 \
    --variant b --video-vib sink --fusion sink_sym \
    --log-path runs/sinksym_smoke/log.jsonl \
    --ckpt-path runs/sinksym_smoke/final.pt \
    --print-every 1 --save-every 10
echo "=== Done $(date) ==="
