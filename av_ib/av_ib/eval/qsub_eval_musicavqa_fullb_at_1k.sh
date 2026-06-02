#!/bin/bash
#PBS -N eval_mavqa_fullb_at_1k
#PBS -l select=1
#PBS -l walltime=05:00:00
#PBS -j oe
#PBS -o eval_musicavqa_fullb_at_1k.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L
echo ""

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"
echo ""

echo "=== Evaluating fullb@1k on Music-AVQA test ==="
python -u av_ib/eval/eval_musicavqa.py \
    --base-model  Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --adapter     results/ckpts/fullb_1k/final \
    --ann-path    /home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-test.json \
    --video-root  /home/aab11336im/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/synthetic \
    --output-dir  results/eval/musicavqa \
    --label       fullb@1k \
    --vllm --tp   1

echo "=== Done: $(date) ==="
