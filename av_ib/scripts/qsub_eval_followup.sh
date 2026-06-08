#!/bin/bash
#PBS -N eval_followup
#PBS -P gae50891
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o eval_followup.qsub.log

set -euo pipefail
cd "$PBS_O_WORKDIR"

OUTFILE="$PBS_O_WORKDIR/runs/eval_followup_out.txt"
mkdir -p "$PBS_O_WORKDIR/runs"
exec > >(tee -a "$OUTFILE") 2>&1
echo "=== Node: $(hostname)  Date: $(date) ==="
nvidia-smi -L

source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate av_ib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANN_PATH="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/avqa-train.json"
VIDEO_ROOT="$HOME/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all"
HELD="--ann-path $ANN_PATH --video-root $VIDEO_ROOT --variant b_topk_nofusion \
      --num-samples 150 --seed 99 --train-seed 42 --train-max-samples 2000 \
      --out-dir runs/heldout_eval --every 10"
E1="--ann-path $ANN_PATH --video-root $VIDEO_ROOT --variant b_topk_nofusion \
    --num-samples 40 --seed 42 --every 10"

# SUBMIT AFTER t1_aux0 / t2_nolora_b7e6 / t2_nolora_b3e5 finish.
# Held-out accuracy+NLL+KL (auto-compares against the gate3 siblings already
# in runs/heldout_eval), then the E1 hallucination probe on each.
# Note: --no-lora checkpoints load fine into a use_lora=True eval model
# because LoRA is zero-initialised (identity) and only VIB keys are present.

for tag in t1_aux0 t2_nolora_b7e6 t2_nolora_b3e5; do
    echo "##### held-out: $tag #####"
    python -u -m av_ib.eval.eval_v6_heldout $HELD \
        --label "$tag" --ckpt-path "runs/$tag/final.pt"
done

for tag in t1_aux0 t2_nolora_b7e6 t2_nolora_b3e5; do
    echo "##### E1 hallucination probe: $tag #####"
    python -u -m av_ib.eval.explain e1 $E1 --ckpt-path "runs/$tag/final.pt"
done

echo "=== Done $(date) ==="
