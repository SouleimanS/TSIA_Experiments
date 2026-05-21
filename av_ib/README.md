# TSIA_Experiments: AV-LLM Hallucination Mitigation via Variational Information Bottleneck

Master's thesis (TSIA) on reducing audio-visual hallucination in large multimodal models
using Variational Information Bottleneck (VIB) regularization on the cross-modal token stream.

**Status (2026-05-21)**: Active experimental phase. Pivoted from AVHBench to MUSIC-AVQA.
12-job noise-robustness sweep launched. v4 (C-MIB) NaN bug found and patched.

## TL;DR for resuming work on a new cluster

1. `git clone https://github.com/SouleimanS/TSIA_Experiments.git && cd TSIA_Experiments/av_ib`
2. Install env: `conda env create -f environment.yml -n av_ib && conda activate av_ib`
3. Mount or download datasets (see "Datasets" section below)
4. Resume training from any checkpoint, or re-run from scratch using qsubs in `scripts/`
5. **All experimental data is in `results/logs_snapshot_20260521_1647/`** (JSONL logs per run)
6. **runs/ directory is NOT in git** — it has 59 GB of checkpoints, see `runs/README.md`

## Thesis hypothesis

Multimodal LLMs hallucinate when one modality's signal dominates a question that
requires cross-modal grounding. We test whether inserting a Variational Information
Bottleneck (VIB) between the multimodal token stream and the LLM forces the model
to compress out modality-specific noise and rely on shared cross-modal signal.

### Current finding (preliminary, ~60% through MUSIC-AVQA training)

On clean MUSIC-AVQA, **VIB hurts**:
- v1 (baseline, no VIB): 0.680 overall val accuracy
- v3 (single VIB): 0.636
- v4 (C-MIB, three stacked VIBs + aux heads): collapsed to NaN at step ~2200 (now patched)

Why: encoders (EVA-ViT, ImageBind) and Q-Formers already aggressively compress raw
inputs to 40 × 4096 tokens. Adding a stochastic bottleneck on already-pre-compressed
clean signals throws away useful information.

**Working hypothesis going forward**: VIB should help under noise, where there is
actual irrelevant signal to filter out. The 12-job noise sweep tests this — train
on noise-augmented MUSIC-AVQA, eval on clean.

## Architecture variants

| Code | What it adds | Trainable params | Loss |
|---|---|---|---|
| **v1** | Baseline | ~311M | `nll` |
| **v2_nb{1,2}** | + cross-modal fusion blocks | ~512M / ~714M | `nll` |
| **v3_b{X}** | + single joint VIB (E-MIB), β=X | ~344M | `nll + β·KL` |
| **v4_b{X}** | + per-modality VIBs + joint VIB + 2 aux LM heads (C-MIB, Mai et al. 2023) | ~674M | `nll + β_v·KL_v + β_a·KL_a + β_j·KL_j + aux_w·(nll_aux_v + nll_aux_a)` |

All variants share: EVA-ViT-G video encoder (frozen), ImageBind audio encoder
(frozen), Vicuna-7B-v0 LLM with LoRA r=16 (only LoRA weights + Q-Formers + bottlenecks/aux
heads trainable). See `av_ib/model/av_model_v{1,2,3,4}.py`.

## Datasets

| Dataset | Used for | Path on this cluster |
|---|---|---|
| AVHBench | Initial hallucination benchmark (abandoned — AV Matching task collapses) | `~/SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0/` |
| MUSIC-AVQA | **Current main experiment** | `~/SOULEIMAN_repo/datasets/MUSIC-AVQA/MUSIC-AVQA/data/json_update/` |
| MUSIC-AVQA videos | Audio + video clips (1866 synthetic + 7422 real = 9288) | `~/SOULEIMAN_repo/datasets/MUSIC-AVQA/videos/all/` |

When resuming on a new cluster, **datasets must be re-downloaded**. Paths are
hard-coded near the top of each `scripts/train_*.py` file (variable `MUSICAVQA_ROOT`
and `VID_ROOT`) — update them for the new cluster.

## Repository layout
av_ib/
├── data/
│   ├── musicavqa.py              # MUSIC-AVQA dataset, prompt rendering, type parsing
│   ├── musicavqa_noisy.py        # Noise-augmented wrapper (gaussian / audio mix)
│   └── test_render_question.py   # 11 unit tests, all passing
├── eval/
│   └── musicavqa_eval.py         # Shared eval helper (41-token vocab, per-modality breakdown)
├── model/
│   ├── av_model_v1.py            # Baseline
│   ├── av_model_v2.py            # + fusion blocks
│   ├── av_model_v3.py            # + single VIB
│   ├── av_model_v4.py            # C-MIB: per-modality VIBs + joint VIB + aux heads
│   ├── bottleneck.py             # VIB primitive (patched 2026-05-21 with logvar clamp)
│   ├── encoders.py               # EVA-ViT + ImageBind wrappers (frozen)
│   ├── qformer.py                # Video + Audio Q-Former (BLIP-2 derivative)
│   └── llm.py                    # Vicuna LoRA wrapper
└── ...
scripts/
├── train_musicavqa_v{1,2,3,4}_3ep.py            # Clean training scripts
├── train_musicavqa_v{1,3,4}noisy.py            # Noise-augmented training (clean eval)
├── eval_musicavqa.py                            # Standalone eval with full 9-bucket breakdown
├── plot_curves.py                               # Plot eval_log.jsonl curves
├── qsub_train_musicavqa.sh                    # PBS submission scripts
├── qsub_train_musicavqa_noisy.sh                # Parameterized; takes env vars
├── launch_noisy_sweep.sh                        # Submits all 12 noisy jobs
└── ...
results/
├── logs_snapshot_20260521_1647/                 # JSONL logs from all runs (~20 MB, IN GIT)
├── avhbench_v.json                             # Earlier AVHBench eval results
└── ...
runs/                                            # NOT IN GIT — see runs/README.md (59 GB of .pt files)

## Current state of running jobs (2026-05-21 ~16:45 JST)

15 jobs running on ABCI rt_HF queue (project `gae50891`):
- **2 clean** (continuing): v1, v2_nb1 — at step ~8500/11970
- **4 v1_noisy** (continuing): 2 Gaussian (σ=0.1, 0.5) + 2 audio-mix (σ=0.1, 0.5) — at step ~1000/11970
- **1 v4 clean** (resubmitted after patch): at step ~500
- **4 v3_noisy** (resubmitted after patch): at step ~400-500
- **4 v4_noisy** (resubmitted after patch): at step ~200-500

Notable: `_prepatch` directories under `runs/` contain pre-VIB-fix attempts including
the v4 NaN crash data. They're kept for diagnostic purposes but not used in final numbers.

## The v4 stability patch

Original `bottleneck.py` had no clamping on `logvar`. In v4's stacked VIB topology
(joint VIB sits on top of per-modality VIBs whose inputs are reparameterized samples),
`logvar` saturated at step 2226 of the clean v4 training, causing `exp(logvar)` to
overflow and `kl_j` to jump from 0.74 to 59039.4 in one step. Downstream gradients
went to NaN, poisoning all weights.

Fix in `av_ib/model/bottleneck.py`:

```python
def forward(self, av_tokens):
    mu = self.fc_mu(av_tokens)
    logvar = self.fc_logvar(av_tokens)
    logvar = logvar.clamp(min=-10.0, max=10.0)  # <-- added
    ...
```

Range `[-10, 10]` gives `exp(logvar) ∈ [4.5e-5, 22026]`. Verified by smoke test
and by patched live runs passing through the previous NaN zone with `kl_j ~ 1.0-1.3`.

## How to read the experimental data

The JSONL logs in `results/logs_snapshot_20260521_1647/` are the experimental record.

```python
import json
from pathlib import Path

SNAP = Path("results/logs_snapshot_20260521_1647")

for run_dir in sorted(SNAP.glob("musicavqa_*")):
    eval_log = run_dir / "eval_log.jsonl"
    if not eval_log.exists():
        continue
    with open(eval_log) as f:
        evals = [json.loads(line) for line in f]
    if not evals:
        continue
    best = max(evals, key=lambda e: e["accuracy"])
    print(f"{run_dir.name:50s}  best_step={best['step']:5d}  acc={best['accuracy']:.4f}")
```

To re-plot accuracy curves:

```bash
python scripts/plot_curves.py v1_3ep v3_3ep_b1e-2_prepatch v4_3ep_b1e-2 --modality --out plot.png
```

(Note: `plot_curves.py` looks under `runs/`, so if `runs/` doesn't exist on the new
cluster, either rsync the JSONL logs from the snapshot back into `runs/`, or modify
the script to read from the snapshot directory.)

## What's next

1. Wait for current 15 jobs to finish (~9 hours total walltime each)
2. Run full-val 9-bucket eval on each best.pt (`scripts/eval_musicavqa.py`)
3. Decision tree:
   - If v3/v4 win under noise → thesis is "VIB helps under input corruption"
   - If they don't → thesis is "VIB doesn't help for AV-LLM hallucination on clean curated benchmarks; here's when and why" (negative result, also defensible)
4. AVHBench transfer eval as OOD test (even with known AV Matching collapse)
5. Write up

## Dependencies

See `environment.yml`. Highlights:
- PyTorch 2.x + CUDA 12.x
- transformers (Vicuna), peft (LoRA)
- timm (ViT), decord (video), librosa (audio)
- ABCI cluster: rt_HF queue (H200 nodes, 8 GPUs/node, billed per node not per GPU)
