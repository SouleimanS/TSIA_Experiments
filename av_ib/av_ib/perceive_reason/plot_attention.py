"""Three-way attention comparison: Original | VIB only | VIB + Perception.

Captures LLM decoder attention over visual tokens under three conditions for
the same clip and checkpoint:

  raw       : provider=None — vanilla Qwen (no VIB, no perception)
  vib_only  : VIB provider, _current_desc=None — VIB but cross-attn is a no-op
  perceive  : VIB provider, _current_desc=description — full perceive+reason

This isolates the incremental effect of the perception context over the VIB
alone. Same localization metrics as plot_ib_attention: norm_entropy and
top5pct_mass.

Helpers _extract_frames, _to_heatmap, _attn_row, _run_forward, and
_localization_metrics are imported from the existing IB-attention module to
avoid code duplication.

Usage:
    python -m av_ib.perceive_reason.plot_attention \\
        --video clip.mp4 --ckpt-path runs/perceive/final.pt \\
        --variant b_topk_nofusion --out-dir runs/perceive_attention
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from av_ib.eval.plot_llm_attention_maps import _extract_frames, _to_heatmap, _attn_row
from av_ib.ib_attention.plot_ib_attention import _localization_metrics, _run_forward


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",         required=True)
    ap.add_argument("--ckpt-path",     required=True,
                    help="Trained AVModelPerceive checkpoint.")
    ap.add_argument("--variant",       default="b_topk_nofusion")
    ap.add_argument("--question",      default="What is happening in this video?")
    ap.add_argument("--out-dir",       default="runs/perceive_attention")
    ap.add_argument("--n-frames",      type=int, default=6)
    ap.add_argument("--last-n-layers", type=int, default=8)
    args = ap.parse_args()

    print("Loading AVModelPerceive onto cuda:0...", flush=True)
    from av_ib.perceive_reason.model import AVModelPerceive
    model = AVModelPerceive(
        use_lora=True,
        variant=args.variant,
        device_map="cuda:0",
    ).eval()
    model.set_sample_noise(False)

    sd = torch.load(args.ckpt_path, map_location="cpu")
    if "trainable_state" in sd:
        sd = sd["trainable_state"]
    own = dict(model.named_parameters())
    n_ok = sum(1 for k, v in sd.items()
               if k in own and not own[k].data.copy_(v.data) is None)
    print(f"  loaded {n_ok}/{len(sd)} params", flush=True)

    # ── Locate decoder layers ─────────────────────────────────────────────────
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

    # ── Prepare inputs ────────────────────────────────────────────────────────
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

    # ── Three conditions ──────────────────────────────────────────────────────
    results = {}

    # Mode 1: raw — vanilla Qwen, no VIB, no perception
    model.qwen._current_provider = None
    model._current_desc = None
    print("Running forward [raw]...", flush=True)
    sal = _run_forward(model, inputs, take_idx, layers, n_heads, q_pos, vis_pos)
    results["raw"] = {"sal": sal, "metrics": _localization_metrics(sal)}
    print(f"  [raw] norm_entropy={results['raw']['metrics']['norm_entropy']:.4f}  "
          f"top5pct_mass={results['raw']['metrics']['top5pct_mass']:.4f}", flush=True)

    # Mode 2: vib_only — VIB provider but _current_desc=None (cross-attn skipped)
    model._current_desc = None
    provider_vib, _, _, _ = model._make_provider()
    model.qwen._current_provider = provider_vib
    print("Running forward [vib_only]...", flush=True)
    try:
        sal = _run_forward(model, inputs, take_idx, layers, n_heads, q_pos, vis_pos)
    finally:
        model.qwen._current_provider = None
    results["vib_only"] = {"sal": sal, "metrics": _localization_metrics(sal)}
    print(f"  [vib_only] norm_entropy={results['vib_only']['metrics']['norm_entropy']:.4f}  "
          f"top5pct_mass={results['vib_only']['metrics']['top5pct_mass']:.4f}", flush=True)

    # Mode 3: perceive — VIB + CrossModalAttention with description context
    print("Generating video description for perception pass...", flush=True)
    desc = model._perceive_video(args.video)
    print(f"  description: {desc[:120]!r}", flush=True)
    model._current_desc = desc
    provider_perceive, _, _, _ = model._make_provider()
    model.qwen._current_provider = provider_perceive
    model._current_desc = desc   # re-stash: provider closure reads it at call time
    print("Running forward [perceive]...", flush=True)
    try:
        sal = _run_forward(model, inputs, take_idx, layers, n_heads, q_pos, vis_pos)
    finally:
        model.qwen._current_provider = None
        model._current_desc = None
    results["perceive"] = {"sal": sal, "metrics": _localization_metrics(sal)}
    print(f"  [perceive] norm_entropy={results['perceive']['metrics']['norm_entropy']:.4f}  "
          f"top5pct_mass={results['perceive']['metrics']['top5pct_mass']:.4f}", flush=True)

    # ── Reshape to (T, H_t, W_t) ─────────────────────────────────────────────
    for mode in results:
        results[mode]["grid"] = results[mode]["sal"][: T * H_t * W_t].reshape(T, H_t, W_t)

    rgb_frames = _extract_frames(args.video, args.n_frames)
    if not rgb_frames:
        raise RuntimeError("Could not extract frames.")
    frame_to_t = ([0] * len(rgb_frames) if T == 1 else
                  [round(i * (T - 1) / max(len(rgb_frames) - 1, 1))
                   for i in range(len(rgb_frames))])

    # ── Plot: rows = frames, cols = [Original | VIB only | VIB + Perception] ─
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rgb_frames)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 3.2, n * 2.6))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Original", "Raw (no VIB)", "VIB only", "VIB + Perception"]

    for r, (frame, t_idx) in enumerate(zip(rgb_frames, frame_to_t)):
        axes[r, 0].imshow(frame)
        axes[r, 1].imshow(_to_heatmap(results["raw"]["grid"][t_idx], frame))
        axes[r, 2].imshow(_to_heatmap(results["vib_only"]["grid"][t_idx], frame))
        axes[r, 3].imshow(_to_heatmap(results["perceive"]["grid"][t_idx], frame))
        for c in range(4):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=11)

    mr = results["raw"]["metrics"]
    mv = results["vib_only"]["metrics"]
    mp = results["perceive"]["metrics"]
    fig.suptitle(
        f"{Path(args.video).stem}   "
        f"norm-entropy  raw={mr['norm_entropy']:.3f} / "
        f"vib={mv['norm_entropy']:.3f} / "
        f"perceive={mp['norm_entropy']:.3f}   "
        f"top5%-mass  raw={mr['top5pct_mass']:.3f} / "
        f"vib={mv['top5pct_mass']:.3f} / "
        f"perceive={mp['top5pct_mass']:.3f}",
        fontsize=9,
    )
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{Path(args.video).stem}_perceive_compare.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}", flush=True)

    import json
    (out_dir / f"{Path(args.video).stem}_perceive_metrics.json").write_text(json.dumps({
        "video": Path(args.video).stem,
        "raw": mr,
        "vib_only": mv,
        "perceive": mp,
        "delta_vib_vs_raw": {
            "norm_entropy": mv["norm_entropy"] - mr["norm_entropy"],
            "top5pct_mass": mv["top5pct_mass"] - mr["top5pct_mass"],
        },
        "delta_perceive_vs_vib": {
            "norm_entropy": mp["norm_entropy"] - mv["norm_entropy"],
            "top5pct_mass": mp["top5pct_mass"] - mv["top5pct_mass"],
        },
    }, indent=2))
    print("Done.", flush=True)


if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
