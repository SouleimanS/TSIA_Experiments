#!/bin/bash
#PBS -N musicavqa_bis_1k
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o musicavqa_bis_1k.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

export FORCE_QWENVL_VIDEO_READER=decord

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"
echo ""

mkdir -p results/ckpts/musicavqa_bis_1k

echo "=== musicavqa_bis_1k — variant b_bis ==="
python -u -m av_ib.train.train_v6 \
    --ann-path    /home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train-synthetic.json \
    --video-root  /home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/raw/MUCIS-AVQA-videos-Synthetic \
    --num-steps   1000 \
    --variant     b_bis \
    --video-vib   sink \
    --audio-vib   norm_topk \
    --fusion      mutual \
    --beta-v      0 \
    --beta-a      0 \
    --beta-j      0 \
    --aux-weight  0.1 \
    --lr          1e-4 \
    --save-every  200 \
    --print-every 10 \
    --log-path    results/ckpts/musicavqa_bis_1k/log.jsonl \
    --ckpt-path   results/ckpts/musicavqa_bis_1k/final.pt

echo "=== Done: $(date) ==="
