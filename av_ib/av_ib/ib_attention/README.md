# Attention maps: with vs without the information bottleneck

Inspired by Figure 6 of Nagrani et al., *Attention Bottlenecks for Multimodal
Fusion* (NeurIPS 2021), which shows that a fusion bottleneck makes attention
**more localized** to sound-source regions.

## What this does

From a **single trained checkpoint**, we run the thinker twice and capture the
LLM's attention from the last prompt token over the visual tokens:

| condition    | splicer provider            | what the LLM sees                |
|--------------|-----------------------------|----------------------------------|
| `without_ib` | `None`                      | raw encoder outputs (vanilla)    |
| `with_ib`    | `model._make_provider()`    | VIB-compressed video/audio       |

Everything else — LoRA weights, encoder, prompt — is identical, so the only
difference is the information bottleneck. Output is a per-clip figure with
columns **[ Original | Without IB | With IB ]** plus a metrics sidecar.

## Caveat on relevance

MBT's bottleneck is an **attention** bottleneck (cross-modal attention routed
through a few fusion tokens). Ours is a **representational** bottleneck (a VIB
on token embeddings). MBT's localization effect is therefore not guaranteed to
transfer. To check the claim rather than assume it, each run reports:

- **`norm_entropy`** = H(attention)/log(N) ∈ [0,1]. Lower ⇒ more localized.
- **`top5pct_mass`** = attention mass in the top 5% of patches. Higher ⇒ more localized.

If the IB localizes attention, `with_ib` should have *lower* norm-entropy and
*higher* top-5%-mass than `without_ib`. If the metrics barely move, that is an
honest negative result — the representational IB does not reshape attention the
way an attention bottleneck does.

## Note

This also corrects an oversight in `av_ib/eval/plot_llm_attention_maps.py`,
which never set a splicer provider — so its "trained" maps used the raw encoder
tokens and the bottleneck was never actually active. This script applies it.

## Run

```bash
qsub scripts/ib_attention/plot_ib_attention.qsub
```

or directly:

```bash
python -m av_ib.ib_attention.plot_ib_attention \
    --video clip.mp4 --ckpt-path runs/avqa/v6b_base_final.pt \
    --variant b_topk_nofusion --out-dir runs/ib_attention --n-frames 6
```

Results land in `results/ib_attention/` (`*_ib_compare.png`, `*_ib_metrics.json`).
Aggregate the JSON sidecars across clips to report mean Δentropy / Δmass.
