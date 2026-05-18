#!/bin/bash
#PBS -P gae50891
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -N avhbench_full
#PBS -j oe
#PBS -o /home/aab11336im/SOULEIMAN_repo/datasets/AVHBench/AVHBench-Align-FT/avhbench_full.log

cd ~/SOULEIMAN_repo/datasets/AVHBench/AVHBench-Align-FT

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

echo "=== Starting full AVHBench eval ==="
date
echo ""

python inference_avhbench.py \
    --cfg-path eval_configs/video_llama_eval_withaudio_stage3.yaml \
    --ckpt models/AVHBench/checkpoint_000002_loss_0.291.pth \
    --video-root ~/SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0/video \
    --qa-json ~/SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0/json/qa.json \
    --output predictions_full_6408.csv \
    --gpu-id 0

echo ""
echo "=== Inference done, scoring ==="
date
echo ""

python score_avhbench.py predictions_full_6408.csv

echo ""
echo "=== All done ==="
date
