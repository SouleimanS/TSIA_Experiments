# perceive_reason — Perceive-then-Reason AV Module

## Architecture

`AVModelPerceive` is a two-pass extension of `AVModelV6` that adds an explicit
perception stage before the bottleneck reasoning stage.

### Pass 1 — Perception (frozen, no grad)

Before each training step or generation call, the model runs a short Qwen3-Omni
forward over the input video to produce a text description of the scene:

> *"Briefly list the main objects and scene visible in this video."*

The description is generated with `max_new_tokens=80`, cached by video path
(LRU, max 4096 entries), and never used to compute gradients. This pass uses
the raw backbone (no VIB provider) to avoid compressing the perception signal.

### Pass 2 — Reasoning (trained)

The standard VIB pipeline runs (SinkAwareVIB on video, audio VIB per variant).
Then the cached description is:

1. Tokenized with the Qwen tokenizer.
2. Embedded with the frozen input-embedding table (`get_input_embeddings()`).
3. Mean-pooled/left as sequence context `(1, L, D)`.
4. Used as keys and values in `CrossModalAttention`, where the queries are the
   concatenated `[video | audio]` VIB tokens `(1, N_v+N_a, D)`.

The attended tokens are then spliced into the LLM's `inputs_embeds` by the
existing `CMIBSplicer` — no changes to the splicer needed.

### CrossModalAttention

A standard multi-head cross-attention with residual connection and LayerNorm.
The `out_proj` is **zero-initialized** so an untrained model is the identity —
the perceive module adds zero perturbation at step 0 and the model can recover
the VIB-only baseline exactly.

All ops are out-of-place to avoid autograd version errors with the VIB modules.

## Trained parameters

- `cross_attn.*` — CrossModalAttention (≈33M params for D=2048, H=8)
- LoRA adapters in the Thinker text model (same as base)
- VIB bottlenecks: `bottleneck_v.*`, `bottleneck_a.*`

The Qwen backbone, audio/vision encoders, and aux heads are frozen.

## Two-pass design

```
video_path
    │
    ├──[Pass 1: frozen]──► Qwen generate ──► description string ──► cache
    │
    └──[Pass 2: trained]─► CMIBSplicer ──► SinkAwareVIB ──► CrossModalAttention
                                                                       │
                                                    desc_embeds (frozen)┘
                                                           │
                                                     z_j spliced into LLM
```

## Attention comparison

`plot_attention.py` produces a four-column comparison per frame:

| Column | Condition |
|--------|-----------|
| Original | Raw video frame |
| Raw (no VIB) | Vanilla Qwen, provider=None |
| VIB only | VIB provider, `_current_desc=None` (cross-attn is identity) |
| VIB + Perception | Full perceive-then-reason pass |

Localization metrics (normalized entropy, top-5%-mass) are reported per
condition and saved as a JSON sidecar for aggregation.

## Usage

### Training
```bash
python -m av_ib.perceive_reason.train \
    --dataset avqa \
    --ann-path /path/to/avqa-train.json \
    --video-root /path/to/videos \
    --variant b_topk_nofusion \
    --num-steps 2000 --lr 1e-4 \
    --beta-v 7e-6 --beta-a 7e-6 \
    --n-perceive-frames 4 \
    --cross-attn-heads 8 --cross-attn-dropout 0.1 \
    --log-path runs/perceive/log.jsonl \
    --ckpt-path runs/perceive/final.pt
```

### Attention plots
```bash
python -m av_ib.perceive_reason.plot_attention \
    --video clip.mp4 --ckpt-path runs/perceive/final.pt \
    --variant b_topk_nofusion --out-dir runs/perceive_attention
```

### PBS/qsub
```bash
qsub scripts/perceive_reason/train.qsub
qsub scripts/perceive_reason/plot_attention.qsub
```