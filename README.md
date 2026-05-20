# TSIA Experiments — Audio-Visual Information Bottleneck

Master's thesis codebase: hallucination mitigation in Audio-Visual LLMs via
Variational Information Bottleneck (VIB) and mutual cross-modal fusion.

See `av_ib/MANIFEST.md` for a file-by-file description of the codebase.

## Architecture
Three variants:

| Variant | Fusion (Φ) | Bottleneck (q_θ) | Notes |
|---|---|---|---|
| v1 | Identity | Identity | Baseline, encoders → Q-Formers → concat → LLM |
| v2 | MutualCrossAttention | Identity | + cross-modal fusion (1 or 2 blocks, configurable) |
| v3 | Identity | VIB | + variational bottleneck after concat (β configurable) |

## Setup on a fresh cluster

### 1. Clone the repo

```bash
git clone -b main https://github.com/SouleimanS/TSIA_Experiments.git
cd TSIA_Experiments
```

### 2. Recreate the conda environment

```bash
conda env create -f av_ib/environment.yml -n av_ib
conda activate av_ib
```

### 3. Download model weights

The AVHBench-Align-FT directory provides the frozen encoders and Vicuna base
model. Required checkpoints (paths assume AVHBench is at
`~/SOULEIMAN_repo/datasets/AVHBench/AVHBench-Align-FT/`):

| File | Source | Size |
|---|---|---|
| `models/eva_vit_g.pth` | https://huggingface.co/QuanSun/EVA-CLIP | ~5 GB |
| `models/imagebind_huge.pth` | https://github.com/facebookresearch/ImageBind | ~5 GB |
| `models/blip2_pretrained_flant5xxl.pth` | https://github.com/salesforce/LAVIS | ~700 MB |
| `models/finetune_vicuna7b_videobranch.pth` | AVHBench release | ~700 MB |
| `models/vicuna-7b-v0/` | Apply delta to LLaMA-1-7B per AVHBench's `apply_delta.py` | ~13 GB |
| `models/bert-base-uncased/` | HuggingFace | ~500 MB |

The paths in `av_ib/av_ib/model/{encoders,qformer,llm}.py` assume those exact
locations. Adjust the `_AVHBENCH_ROOT` constants if you put the files elsewhere.

### 4. Download datasets

- **AVQA** (training): http://mn.cs.tsinghua.edu.cn/avqa/ → videos to
  `~/SOULEIMAN_repo/datasets/AVQA/videos/Train/`, annotations to
  `~/SOULEIMAN_repo/datasets/AVQA/AVQA/AVQA_dataset/`.
- **AVHBench** (eval): https://github.com/kaist-ami/AVHBench → videos and
  annotations to `~/SOULEIMAN_repo/datasets/AVHBench/`.

### 5. Verify the environment

```bash
cd av_ib
conda activate av_ib
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python scripts/test_av_model_v1.py     # end-to-end forward pass smoke test
```

## Running experiments

### Single training job

```bash
cd av_ib

# v1 baseline (no fusion, no VIB)
qsub -P <project> -q <queue> scripts/qsub_train_avqa_3ep.sh

# v2 with depth N
qsub -P <project> -q <queue> scripts/qsub_train_v2_nb1.sh   # 1 cross-attention block
qsub -P <project> -q <queue> scripts/qsub_train_v2_nb2.sh   # 2 blocks

# v3 with β
qsub -P <project> -q <queue> scripts/qsub_train_v3_b1e-3.sh
```

All training scripts now accept `--beta`, `--n-blocks`, `--output-dir`, and
`--smoke-steps` CLI args. See `av_ib/scripts/train_avqa_v{2,3}_3ep.py`.

### Smoke-test all configs in one job (~5 min on one H200)

```bash
qsub -P <project> -q <queue> scripts/qsub_smoke_all.sh
```

Runs 3 training steps per config (2 v2 + 5 v3) to verify the full pipeline
works before launching the real overnight sweep.

### Evaluating a checkpoint

```bash
python scripts/eval_avhbench.py --variant v3 --checkpoint runs/avqa_v3_3ep_b1e-3/best.pt
```

Writes `results/avhbench_v3.{json,csv}` with per-task accuracy, confusion
matrix, and per-item predictions.

## Cluster notes (ABCI)

- Queue: `rt_HF`. Submit with `qsub -P gae50891 -q rt_HF <wrapper>`.
- Each job gets a full node (8× H200, 192 cores, 1920 GB RAM). Training uses
  one GPU per job; the rest sit idle but the allocation can't be subdivided.
- Walltime: 6h is comfortable for 3-epoch training on AVQA (actual ~2.5h).

## What's NOT in this repo

- Model weights (frozen encoders, Vicuna, etc.). See "Download model weights" above.
- Dataset videos and annotations. See "Download datasets" above.
- Training outputs (`runs/`). Each run lands in `av_ib/runs/<name>/` locally;
  intentionally gitignored — they're reproducible from this code + the data.
- Conda env binaries — only the spec is committed (`av_ib/environment.yml`).

## Related forks (your modifications to baseline repos)

Tonight's work depends on three baseline repos that have been forked under
SouleimanS with small patches:

- https://github.com/SouleimanS/VideoLLaMA2 (branch `audio_visual`) — AVHBenchDataset
- https://github.com/SouleimanS/PandaGPT (branch `main`) — do_sample fix + audio target_length
- https://github.com/SouleimanS/LLaMA-Adapter (branch `main`) — torch._six → torch.inf

Clone these alongside if you want to re-run the baseline evaluations on
AVHBench. They aren't required to train or eval the v1/v2/v3 variants.
