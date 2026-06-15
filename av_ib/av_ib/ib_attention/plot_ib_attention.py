"""Attention maps WITH vs WITHOUT the information bottleneck.

Replicates the comparison in Figure 6 of Nagrani et al., "Attention Bottlenecks
for Multimodal Fusion" (NeurIPS 2021): for one clip we overlay the original
frame, the attention map for the *vanilla* (no-bottleneck) model, and the
attention map for the *bottleneck* model. The paper's claim is that the
bottleneck makes attention more **localized** to sound-source regions.

Important difference from MBT: MBT's bottleneck is an *attention* bottleneck
(cross-modal attention is forced through a few fusion tokens). Ours is a
*representational* bottleneck (a VIB on the video/audio token embeddings before
they are spliced into the LLM). So this is a test of whether MBT's localization
effect also appears for a representational IB — not a guaranteed replication.

The two conditions are produced from the SAME trained checkpoint, differing
only in the splicer provider:

  without_ib : provider = None  -> the LLM consumes the raw encoder outputs
               that Qwen itself injects (vanilla path, no compression).
  with_ib    : provider = model._make_provider() -> the video/audio embeddings
               are passed through the trained VIB before splicing.

This isolates the bottleneck's effect on attention with everything else
(LoRA weights, encoder, prompt) held fixed.

Beyond the heatmaps we report two localization metrics over the visual-token
attention distribution, so the MBT claim can be checked quantitatively rather
than by eye:

  norm_entropy : H(p)/log(N) in [0,1]. Lower = more localized.
  top5pct_mass : fraction of attention mass in the top 5% of patches.
                 Higher = more localized.

Usage:
    python -m av_ib.ib_attention.plot_ib_attention \\
        --video clip.mp4 --ckpt-path runs/avqa/v6b_base_final.pt \\
        --variant b_topk_nofusion --out-dir runs/ib_attention --n-frames 6
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

# Reuse the proven helpers (frame extraction, heatmap blend, GQA attention row)
from av_ib.eval.plot_llm_attention_maps import (
    _extract_frames, _to_heatmap, _attn_row,
)


# ─────────────────────────────────────────────────────────────────────────────
def _localization_metrics(sal: np.ndarray) -> dict:
    """Localization of an attention distribution over visual tokens.

    sal: (N,) non-negative saliency. Returns normalized entropy and the
    fraction of mass in the top 5% of patches.
    """
    p = sal.astype(np.float64)
    p = np.clip(p, 0, None)
    s = p.sum()
    if s <= 0:
        return {"norm_entropy": float("nan"), "top5pct_mass": float("nan")}
    p = p / s
    N = p.size
    H = -(p * np.log(p + 1e-12)).sum()
    norm_entropy = float(H / math.log(N)) if N > 1 else 0.0
    k = max(1, int(round(0.05 * N)))
    top_mass = float(np.sort(p)[-k:].sum())
    return {"norm_entropy": norm_entropy, "top5pct_mass": top_mass}


def _run_forward(model, inputs, take_idx, layers, n_heads,
                 q_pos, vis_pos) -> np.ndarray:
    """One thinker forward with hooks; returns mean saliency over visual tokens.

    Assumes the splicer provider has already been set on model.qwen.
    """
    q_store: dict[int, torch.Tensor] = {}
    k_store: dict[int, torch.Tensor] = {}
    hooks = []

    def _mk(store, idx):
        def hook(module, inp, out):
            store[idx] = out.detach()
        return hook

    for i in take_idx:
        sa = layers[i].self_attn
        hooks.append(sa.q_proj.register_forward_hook(_mk(q_store, i)))
        hooks.append(sa.k_proj.register_forward_hook(_mk(k_store, i)))

    try:
        with torch.no_grad():
            model.qwen.model.thinker(**inputs, use_audio_in_video=True)
    finally:
        for h in hooks:
            h.remove()

    sal, count = None, 0
    for i in take_idx:
        if i in q_store and i in k_store:
            row = _attn_row(q_store[i], k_store[i], q_pos, vis_pos, n_heads)
            sal = row if sal is None else sal + row
            count += 1
    if sal is None:
        raise RuntimeError("No attention captured.")
    return sal / count


def _run_forward_grad(model, inputs, layers, q_pos, vis_pos) -> np.ndarray:
    """Gradient x input saliency over visual tokens.

    Why this beats raw attention: attention maps are dominated by sinks and
    positional artifacts. The GRADIENT of the answer logit w.r.t. each visual
    token measures how much that token *causally* drives the prediction — it
    naturally concentrates on the object regions that matter and ignores sinks.
    This is what produces clean, MBT-style maps.

    We install a pre-hook on the first decoder layer that swaps the incoming
    embeddings for a leaf tensor, run a real (grad-enabled) forward, take the
    model's own top answer token at q_pos, and backprop to the leaf.
    """
    leaf_box: dict[str, torch.Tensor] = {}

    def pre_hook(module, args):
        x = args[0].detach().requires_grad_(True)
        leaf_box["x"] = x
        return (x,) + args[1:]

    h = layers[0].register_forward_pre_hook(pre_hook)
    try:
        out = model.qwen.model.thinker(**inputs, use_audio_in_video=True,
                                       use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        row = logits[0, q_pos].float()
        target = int(row.argmax().item())
        loss = -torch.log_softmax(row, dim=-1)[target]
        leaf = leaf_box["x"]
        grad = torch.autograd.grad(loss, leaf, retain_graph=False)[0]  # (1, L, D)
    finally:
        h.remove()

    # grad x input, summed over hidden dim, positive part = supporting evidence
    sal_full = (grad[0] * leaf.detach()[0]).sum(dim=-1)          # (L,)
    sal_full = torch.relu(sal_full)
    sal = sal_full[vis_pos.to(sal_full.device)].float().cpu().numpy()
    return sal


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",         required=True)
    ap.add_argument("--ckpt-path",     required=True,
                    help="Trained checkpoint (the bottleneck model).")
    ap.add_argument("--variant",       default="b_topk_nofusion")
    ap.add_argument("--question",      default="What is happening in this video?")
    ap.add_argument("--out-dir",       default="runs/ib_attention")
    ap.add_argument("--n-frames",      type=int, default=6)
    ap.add_argument("--last-n-layers", type=int, default=8)
    ap.add_argument("--method",        default="grad", choices=["grad", "attention"],
                    help="grad = gradient x input saliency (clean, causal); "
                         "attention = raw decoder attention (noisier).")
    args = ap.parse_args()

    print("Loading model onto cuda:0...", flush=True)
    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(use_lora=True, variant=args.variant,
                      device_map="cuda:0").eval()
    model.set_sample_noise(False)   # deterministic z = mu for a fair comparison

    sd = torch.load(args.ckpt_path, map_location="cpu")
    if "trainable_state" in sd:
        sd = sd["trainable_state"]
    own = dict(model.named_parameters())
    n_ok = sum(1 for k, v in sd.items()
               if k in own and not own[k].data.copy_(v.data) is None)
    print(f"  loaded {n_ok}/{len(sd)} params", flush=True)

    # locate decoder layers
    thinker_model = model.qwen.model.thinker.model
    if hasattr(thinker_model, "get_base_model"):
        inner = thinker_model.get_base_model()
    elif hasattr(thinker_model, "base_model"):
        inner = thinker_model.base_model
    else:
        inner = thinker_model
    layers = inner.model.layers if hasattr(inner, "model") else inner.layers
    n_layers = len(layers)
    take_idx = list(range(max(0, n_layers - args.last_n_layers), n_layers))

    q_proj_0 = layers[take_idx[0]].self_attn.q_proj
    D_out = q_proj_0.weight.shape[0]
    for hd in (128, 64, 256, 32):
        if D_out % hd == 0:
            n_heads = D_out // hd
            break
    print(f"  {n_layers} layers, hooking {take_idx}, n_heads={n_heads}", flush=True)

    if args.method == "grad":
        # gradient saliency needs a backward pass; gradient checkpointing keeps
        # the 30B activations within one 80 GB GPU. Frozen params store no grad.
        try:
            model.qwen.model.thinker.gradient_checkpointing_enable()
            print("  gradient checkpointing enabled", flush=True)
        except Exception as e:
            print(f"  (grad checkpointing unavailable: {e})", flush=True)

    # inputs
    print(f"Preparing inputs for {Path(args.video).name}...", flush=True)
    inputs, prompt_len = model.qwen._prep_inputs(args.video, args.question, answer=None)
    tok = model.qwen.tokenizer
    video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    ids = inputs["input_ids"][0]
    vis_mask = (ids == video_pad_id) | (ids == image_pad_id)
    vis_pos = vis_mask.nonzero(as_tuple=True)[0]
    n_vis = vis_pos.numel()
    if n_vis == 0:
        raise RuntimeError("No visual tokens found.")

    _vg = inputs.get("video_grid_thw")
    _ig = inputs.get("image_grid_thw")
    grid_thw = _vg if _vg is not None else _ig
    T, H_p, W_p = int(grid_thw[0, 0]), int(grid_thw[0, 1]), int(grid_thw[0, 2])
    H_t, W_t = H_p // 2, W_p // 2
    if T * H_t * W_t != n_vis:
        for sm in (1, 2, 4):
            H_t, W_t = H_p // sm, W_p // sm
            if T * H_t * W_t == n_vis:
                break
        else:
            side = int(math.isqrt(n_vis // max(T, 1)))
            H_t = W_t = side
    print(f"  grid: T={T}, H_t={H_t}, W_t={W_t}, n_vis={n_vis}", flush=True)
    q_pos = prompt_len - 1

    # ── two conditions ───────────────────────────────────────────────────────
    results = {}
    for mode in ("without_ib", "with_ib"):
        if mode == "without_ib":
            model.qwen._current_provider = None        # vanilla Qwen splice
        else:
            provider, _, _, _ = model._make_provider()  # trained VIB
            model.qwen._current_provider = provider
        print(f"Running forward [{mode}] method={args.method}...", flush=True)
        try:
            if args.method == "grad":
                sal = _run_forward_grad(model, inputs, layers, q_pos, vis_pos)
            else:
                sal = _run_forward(model, inputs, take_idx, layers, n_heads,
                                   q_pos, vis_pos)
        finally:
            model.qwen._current_provider = None
        metrics = _localization_metrics(sal)
        results[mode] = {"sal": sal, "metrics": metrics}
        print(f"  [{mode}] norm_entropy={metrics['norm_entropy']:.4f}  "
              f"top5pct_mass={metrics['top5pct_mass']:.4f}", flush=True)

    # ── reshape to (T, H_t, W_t) ─────────────────────────────────────────────
    for mode in results:
        results[mode]["grid"] = results[mode]["sal"][: T * H_t * W_t].reshape(T, H_t, W_t)

    rgb_frames = _extract_frames(args.video, args.n_frames)
    if not rgb_frames:
        raise RuntimeError("Could not extract frames.")
    frame_to_t = ([0] * len(rgb_frames) if T == 1 else
                  [round(i * (T - 1) / max(len(rgb_frames) - 1, 1))
                   for i in range(len(rgb_frames))])

    # ── plot: rows = frames, cols = [Original | Without IB | With IB] ─────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rgb_frames)
    fig, axes = plt.subplots(n, 3, figsize=(3 * 3.2, n * 2.6))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Original",
                  "Without IB",
                  "With IB"]
    for r, (frame, t_idx) in enumerate(zip(rgb_frames, frame_to_t)):
        axes[r, 0].imshow(frame)
        axes[r, 1].imshow(_to_heatmap(results["without_ib"]["grid"][t_idx], frame))
        axes[r, 2].imshow(_to_heatmap(results["with_ib"]["grid"][t_idx], frame))
        for c in range(3):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=11)

    m0, m1 = results["without_ib"]["metrics"], results["with_ib"]["metrics"]
    fig.suptitle(
        f"{Path(args.video).stem}   "
        f"norm-entropy  without={m0['norm_entropy']:.3f} / with={m1['norm_entropy']:.3f}   "
        f"top5%-mass  without={m0['top5pct_mass']:.3f} / with={m1['top5pct_mass']:.3f}",
        fontsize=10,
    )
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{Path(args.video).stem}_ib_compare.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}", flush=True)

    # also dump metrics as a sidecar for aggregation
    import json
    (out_dir / f"{Path(args.video).stem}_ib_metrics.json").write_text(json.dumps({
        "video": Path(args.video).stem,
        "without_ib": m0,
        "with_ib": m1,
        "delta_norm_entropy": m1["norm_entropy"] - m0["norm_entropy"],
        "delta_top5pct_mass": m1["top5pct_mass"] - m0["top5pct_mass"],
    }, indent=2))
    print("Done.", flush=True)


if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
