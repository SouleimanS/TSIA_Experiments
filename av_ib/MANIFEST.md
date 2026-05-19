# av_ib repository manifest

This is the single source of truth for what every file does. For each
`.py` file the docstring at the top of the file is the canonical
description; this manifest mirrors it for at-a-glance reading.

## Package: `av_ib/`

The core library. Imported by every training and evaluation script.

### `av_ib/data/`

| File | Description |
|---|---|
| `avqa.py` | AVQA Dataset class. Reads videos + yes/no questions from a JSON annotation file. Honors a per-item `video_root` field so we can mix sources (AVQA + AVHBench) in one dataset. |
| `dummy.py` | Random-tensor dataset used for fast smoke tests of the training loop. Also provides `av_collate` (the collation function every other dataset uses). |
| `synthetic_video.py` | Dataset that plants a known visual signal so we can verify the architecture actually uses video input. |
| `synthetic_audio.py` | Same idea, audio-only signal. Sanity check that the audio branch works. |
| `avqa_paraphrase.py` | One-shot script: GPT-4o-mini paraphrase of AVQA multi-choice questions. |
| `avqa_reformulate.py` | One-shot script: turn AVQA multi-choice into balanced yes/no items in AVHBench schema. |

### `av_ib/model/`

| File | Description |
|---|---|
| `encoders.py` | Frozen video and audio encoders. Wraps AVHBench's EVA-ViT-G (video) and ImageBind (audio) loaders. 0 trainable params. |
| `qformer.py` | Trainable Q-Formers: VideoQFormer (~240M params, BLIP-2 init) and AudioQFormer (~54M params, AVHBench init). Bridges frozen encoders to LLM token space. |
| `llm.py` | Vicuna-7B-v0 wrapper with LoRA (~17M trainable). Builds prompts, accepts AV soft-prompt tokens via `inputs_embeds`. |
| `fusion.py` | Cross-modal fusion modules. `Identity` (passthrough for v1) and `MutualCrossAttention` (~200M trainable for v2+). |
| `bottleneck.py` | Information bottleneck modules. `Identity`, `VIB` (~34M trainable, joint AV bottleneck, used by v3), `PerModalityVIB` (planned for v5+). |
| `av_model_v1.py` | Variant 1 architecture: encoders → Q-Formers → concat → LLM. No fusion, no VIB. The baseline. |
| `av_model_v2.py` | Variant 2: v1 + MutualCrossAttention between Q-Formers and concat. |
| `av_model_v3.py` | Variant 3: v1 + VIB bottleneck after concat. Returns `(nll, kl)` from forward_train so the training loop adds `β·KL`. |

### `av_ib/eval/`

| File | Description |
|---|---|
| `avhbench.py` | Shared eval helpers: `_load_video`, `_load_audio`, `_build_prompt`, and a `run_eval` that writes a predictions CSV. Used by every training script's periodic eval. |

## Scripts: `scripts/`

### Training (single-GPU)

| Script | What it trains | Data | Output dir |
|---|---|---|---|
| `train_avqa_1epoch.py` | v1, 1 epoch | AVQA yes/no | `runs/avqa_v1_1ep/` |
| `train_avqa_3ep.py` | v1, 3 epochs | AVQA yes/no | `runs/avqa_v1_3ep/` |
| `train_avqa_v2_3ep.py` | v2 (Fusion), 3 epochs | AVQA yes/no | `runs/avqa_v2_3ep/` |
| `train_avqa_v3_3ep.py` | v3 (VIB), 3 epochs | AVQA yes/no | `runs/avqa_v3_3ep/` |
| `train_combined_v1_5ep.py` | v1, 5 epochs | AVQA + AVHBench-split-train | `runs/combined_v1_5ep/` |
| `train_combined_v2_5ep.py` | v2, 5 epochs | AVQA + AVHBench-split-train | `runs/combined_v2_5ep/` |
| `train_combined_v3_5ep.py` | v3, 5 epochs | AVQA + AVHBench-split-train | `runs/combined_v3_5ep/` |

### Training (multi-GPU, deprecated)

| Script | Notes |
|---|---|
| `train_avqa_3ep_ddp.py` | DDP version of v1 3-epoch. Abandoned because ABCI 3.0 queues only allow 1 GPU per job. Kept for reference. |

### Evaluation

| Script | Purpose | Output |
|---|---|---|
| `eval_avhbench.py` | Eval any variant on AVHBench test set. Takes `--variant v1\|v2\|v3`. Computes accuracy/precision/recall/F1/Yes% per task and overall. | `results/avhbench_{variant}.{json,csv}` |
| `eval_best_full_val.py` | Eval `runs/avqa_v1_1ep/best.pt` on full AVQA val (1238 items). | Console |
| `eval_v1.py` | Quick smoke eval of v1 on first 50 AVHBench items. | Console |
| `verify_eval_function.py` | Sanity-check the new `full_val_eval` function against the known-good 1-epoch checkpoint (should report ~0.68). | Console |

### Data preparation

| Script | Purpose | Output |
|---|---|---|
| `split_avhbench.py` | Split AVHBench yes/no items 80/20 by video. Adds `video_root` field. | `data/avhbench_split_train.json`, `data/avhbench_split_test.json` |
| `build_combined_train.py` | Concatenate AVQA-train + AVHBench-split-train. | `data/combined_train.json` |

### Smoke tests

| Script | What it tests |
|---|---|
| `smoke_train.py` | Full training step on dummy data (4 items, 5 steps). Catches stupid integration bugs. |
| `smoke_train_avqa.py` | Same on real AVQA data (4 items, 2 steps). |
| `test_encoders.py` | EVA-ViT + ImageBind forward pass produces the expected shapes. |
| `test_qformer.py` | Video and audio Q-Formers produce the expected shapes from encoder features. |
| `test_llm.py` | Vicuna + LoRA loads, generates, and computes loss with AV soft-prompts. |
| `test_av_model_v1.py` | End-to-end smoke test on real data: encoders → Q-Formers → LLM → loss. |
| `test_avqa_dataset.py` | AVQADataset returns the expected dict per item. |
| `test_synthetic_video.py` | Architecture can learn a planted video signal (sanity for the video branch). |
| `test_synthetic_audio.py` | Architecture can learn a planted audio signal (sanity for the audio branch). |
| `test_reformulate.py` | Test AVQA-to-yes/no reformulation with hand-crafted mock data, no API calls. |

### Qsub wrappers (PBS submission scripts)

For every training/eval script there is a `qsub_*.sh` PBS wrapper that:
- Sets PBS directives (walltime, GPU count, output log path)
- Activates the `av_ib` conda env
- Runs the corresponding python script

Naming convention: `qsub_<python_script_basename>.sh`. Submit with
`qsub -P gae50891 -q rt_HF <wrapper>`. Each writes to a `*.qsub.log`
matching the wrapper name.

## Outputs

### `runs/` (gitignored)

One directory per training run. Contents per run:
- `train_log.jsonl` — one line per training step (step, loss, grad norm, elapsed)
- `eval_log.jsonl` — one line per periodic eval (step, accuracy, confusion matrix, per-task breakdown)
- `best.pt` — checkpoint with the best val accuracy seen so far
- `final.pt` — checkpoint at end of training

### `results/` (json tracked, csv gitignored)

| File | What it is | Date |
|---|---|---|
| `avhbench_v1.json` | v1 checkpoint (AVQA-trained 80.5%) evaluated on AVHBench test 5302 items | 2026-05-19 |
| `avhbench_v2.json` | v2 (Fusion) checkpoint evaluated on AVHBench test | 2026-05-19 |
| `avhbench_v3.json` | v3 (VIB) checkpoint evaluated on AVHBench test | 2026-05-19 |
| `avhbench_v*.csv` | Per-item predictions for inspection (gitignored, ~470KB each) | 2026-05-19 |

### `data/`

| File | What it is | Tracked? |
|---|---|---|
| `avhbench_split_train.json` | 80% of AVHBench yes/no items, ~4240 items, with `video_root` field | Yes |
| `avhbench_split_test.json` | 20% held-out, ~1060 items | Yes |
| `combined_train.json` | AVQA-train ∪ AVHBench-split-train, ~15250 items | No (regenerable) |

## Current results summary

(See `results/avhbench_*.json` for the full metrics.)

| Variant | AVQA-val acc | AVHBench test acc (overall) | Notes |
|---|---|---|---|
| v1 (no F, no V) | 0.805 | 0.572 | AVQA-trained baseline |
| v2 (Fusion) | 0.669 | 0.538 | AVQA-trained |
| v3 (VIB β=1e-3) | 0.686 | 0.504 | AVQA-trained, predicts mostly "No" on AVHBench |

Combined-trained variants (AVQA + AVHBench-split-train, 5 epochs) are
running as of 2026-05-19. Results will appear in `results/avhbench_combined_*.json`
when complete.

## How to add a new variant

Roughly:

1. Add module to `av_ib/model/` (e.g., `av_model_v4.py`).
2. Copy `scripts/train_combined_v1_5ep.py` to `train_combined_v4_5ep.py`, swap the import.
3. Copy `scripts/qsub_train_combined_v1_5ep.sh` to `qsub_train_combined_v4_5ep.sh`, change the `-N`, `-o`, and python script lines.
4. Submit: `qsub -P gae50891 -q rt_HF scripts/qsub_train_combined_v4_5ep.sh`.
5. Evaluate when done with a copy of `eval_avhbench.py` (or just add the variant to its `--variant` choices).
6. Add a row to the "Current results summary" table above.