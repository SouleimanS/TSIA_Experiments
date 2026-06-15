# token_analysis

Answers: **what happens to video and audio token weights (norms) and LLM attention when the VIB is active vs not?**

## What it does

For each clip, runs two forward passes through the trained checkpoint:

- **`raw`**: `provider=None` — LLM sees the raw Qwen encoder outputs.
- **`vib`**: `provider=model._make_provider()` — LLM sees VIB-compressed embeddings.

Per pass, two hooks capture:

1. **Hook A — embeddings** at `layers[0].input_layernorm` (first LLM decoder layer):
   - `vid_embeds` `(N_v, D)` — video token rows
   - `aud_embeds` `(N_a, D)` — audio token rows

2. **Hook B — attention** at `q_proj` / `k_proj` of the last 8 decoder layers:
   - `attn_full` `(L,)` — GQA-aware softmax attention from the last prompt token over all token positions, averaged over heads and layers.

From these, per-clip metrics are computed:

| Metric | Description |
|--------|-------------|
| `vid_norm_raw / vib` | Mean L2 norm of video token embeddings |
| `aud_norm_raw / vib` | Mean L2 norm of audio token embeddings |
| `vid_cos_sim` | Per-token cosine similarity between raw and VIB video embeddings |
| `aud_cos_sim` | Per-token cosine similarity between raw and VIB audio embeddings |
| `attn_video_raw / vib` | Fraction of total attention mass on video tokens |
| `attn_audio_raw / vib` | Fraction of total attention mass on audio tokens |
| `attn_text_raw / vib` | Fraction of total attention mass on text tokens |

## Outputs

- `{stem}_token_analysis.json` — per-clip metrics
- `{stem}_token_analysis.png` — per-clip figure (attention heatmap + bar chart)
- `token_analysis_summary.json` — means across all clips
- `token_analysis_summary.png` — 4-subplot summary figure

## Usage

```bash
# Single clip
python -m av_ib.token_analysis.analyse \
    --video clip.mp4 \
    --ckpt-path runs/avqa/v6b_base_final.pt \
    --variant b_topk_nofusion \
    --out-dir results/token_analysis

# Directory of clips
python -m av_ib.token_analysis.analyse \
    --video-dir /data/avhbench \
    --n-clips 10 \
    --ckpt-path runs/avqa/v6b_base_final.pt \
    --out-dir results/token_analysis \
    --dataset myclip1 AVHBench \
    --dataset myclip2 AVQA
```

## qsub

```bash
qsub scripts/token_analysis/run_analysis.qsub
```

## Splicer mask extraction

Video/audio token positions are recovered by temporarily patching `_clear_per_call_state` on the splicer to a no-op, running the forward, reading `_splicer._video_mask` / `_splicer._audio_mask`, then restoring. Masks may be 2D `(B, L)` or 3D `(B, L, D)`; the code handles both.
