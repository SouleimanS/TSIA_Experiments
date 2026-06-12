"""Attention map visualisation for the Qwen3-Omni visual encoder.

Usage — two-step:

  Step 1: discover layer names (run once, prints module tree):
      python -m av_ib.eval.plot_attention_maps --video path/to/clip.mp4 --discover

  Step 2: plot attention maps for sampled frames:
      python -m av_ib.eval.plot_attention_maps \
          --video path/to/clip.mp4 \
          --out-dir runs/attention_maps \
          [--ckpt-path runs/avqa/v6b_base_final.pt] \
          [--variant b_topk_nofusion] \
          [--n-frames 6] \
          [--layer-idx -1]        # which attention layer (-1 = last)

The script hooks into the visual encoder's self-attention layers, extracts the
attention weights from the [CLS] token (or mean-pooled query if no CLS), rolls
them out across layers (Attention Rollout), resizes to the original frame
resolution, and saves one heatmap image per sampled frame.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
    """Return n_frames uniformly sampled RGB frames as uint8 arrays (H,W,3)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _find_attn_layers(module: nn.Module, prefix: str = "") -> list[tuple[str, nn.Module]]:
    """Walk module tree and return (name, layer) for anything that looks like
    a self-attention layer (has q_proj / qkv / in_proj)."""
    hits = []
    for name, child in module.named_modules():
        cname = type(child).__name__.lower()
        attrs = set(n for n, _ in child.named_children())
        if any(k in attrs for k in ("q_proj", "qkv", "in_proj", "query")):
            hits.append((name, child))
    return hits


def _rollout(attn_weights: list[np.ndarray], add_residual: bool = True) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020).

    attn_weights: list of (n_heads, N, N) arrays, one per layer.
    Returns: (N,) relevance scores for the CLS token (index 0).
    """
    result = np.eye(attn_weights[0].shape[-1])
    for a in attn_weights:
        a_mean = a.mean(axis=0)  # (N, N) — average over heads
        if add_residual:
            a_mean = a_mean + np.eye(a_mean.shape[0])
            a_mean = a_mean / a_mean.sum(axis=-1, keepdims=True)
        result = a_mean @ result
    # Row 0 is the CLS token's attention over all patch tokens
    return result[0]  # (N,)


def _to_heatmap(scores: np.ndarray, h_patches: int, w_patches: int,
                frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay attention heatmap on frame. Returns uint8 RGB image."""
    import cv2
    n = h_patches * w_patches
    patch_scores = scores[1:n + 1] if len(scores) > n else scores[:n]
    grid = patch_scores.reshape(h_patches, w_patches)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    H, W = frame.shape[:2]
    heat = cv2.resize(grid.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    heat_uint8 = (heat * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    blended = (alpha * colormap_rgb + (1 - alpha) * frame).astype(np.uint8)
    return blended


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        required=True)
    ap.add_argument("--out-dir",      default="runs/attention_maps")
    ap.add_argument("--ckpt-path",    default=None)
    ap.add_argument("--variant",      default="b_topk_nofusion")
    ap.add_argument("--n-frames",     type=int, default=6)
    ap.add_argument("--layer-idx",    type=int, default=-1,
                    help="Which attention layer to use (-1=last, or 0..N). "
                         "Pass 'all' via --rollout to aggregate all layers.")
    ap.add_argument("--rollout",      action="store_true",
                    help="Use attention rollout (aggregate all layers). "
                         "Overrides --layer-idx.")
    ap.add_argument("--discover",     action="store_true",
                    help="Print visual encoder module tree and exit.")
    args = ap.parse_args()

    # ── build model ──
    print("Loading model...", flush=True)
    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(use_lora=bool(args.ckpt_path), variant=args.variant).eval()
    model.set_sample_noise(False)

    if args.ckpt_path:
        sd = torch.load(args.ckpt_path, map_location="cpu")
        if "trainable_state" in sd:
            sd = sd["trainable_state"]
        own = dict(model.named_parameters())
        n_ok = sum(1 for k, v in sd.items()
                   if k in own and not own[k].data.copy_(v.data) is None)
        print(f"  loaded {n_ok}/{len(sd)} params", flush=True)

    vis_enc = model.qwen.visual_encoder

    # ── discover mode ──
    if args.discover:
        print("\n=== Visual encoder module tree ===")
        for name, mod in vis_enc.named_modules():
            if name:
                indent = "  " * name.count(".")
                print(f"{indent}{name}: {type(mod).__name__}")
        print("\n=== Attention-like layers found ===")
        for name, _ in _find_attn_layers(vis_enc):
            print(f"  {name}")
        return

    # ── register hooks ──
    attn_layers = _find_attn_layers(vis_enc)
    if not attn_layers:
        raise RuntimeError(
            "No attention layers found. Run with --discover to inspect the module tree.")

    print(f"Found {len(attn_layers)} attention layers in visual encoder.", flush=True)

    _captured: list[Optional[torch.Tensor]] = []  # one entry per forward call

    def _make_hook(storage: list):
        def hook(module, input, output):
            # output may be (attn_output, attn_weights) or just attn_output
            if isinstance(output, tuple) and len(output) >= 2:
                w = output[1]  # (B, heads, N, N)
                if w is not None:
                    storage.append(w.detach().cpu().float().numpy())
        return hook

    all_layer_storage: list[list] = [[] for _ in attn_layers]
    hooks = []
    for i, (name, layer) in enumerate(attn_layers):
        # Need attention weights: patch nn.MultiheadAttention to return them
        if hasattr(layer, "forward"):
            h = layer.register_forward_hook(_make_hook(all_layer_storage[i]))
            hooks.append(h)

    # ── extract frames ──
    frames = _extract_frames(args.video, args.n_frames)
    if not frames:
        raise RuntimeError(f"Could not extract frames from {args.video}")
    print(f"Extracted {len(frames)} frames.", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(args.video).stem

    # ── run one frame at a time through the visual encoder ──
    # We use the processor to prepare the video input, then run only the
    # visual encoder with a forward hook, bypassing the full model.

    from av_ib.model.av_model_v6 import AVModelV6  # already imported
    tok = model.qwen.tokenizer
    proc = model.qwen.processor

    for fi, frame_rgb in enumerate(frames):
        # Clear storage
        for s in all_layer_storage:
            s.clear()

        # Prepare a minimal conversation with just this frame as an image
        import tempfile, cv2
        import PIL.Image as PILImage
        pil_img = PILImage.fromarray(frame_rgb)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tmp_path = tf.name
        pil_img.save(tmp_path)

        # Build a minimal conversation that feeds just the image
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": tmp_path},
                {"type": "text", "text": "Describe this image."},
            ]},
        ]
        try:
            text = proc.apply_chat_template(conversation,
                                            add_generation_prompt=True,
                                            tokenize=False)
            from qwen_omni_utils import process_mm_info
            _, images_in, _ = process_mm_info(conversation, use_audio_in_video=False)
            inputs = proc(text=text, images=images_in, return_tensors="pt", padding=True)

            first_device = next(vis_enc.parameters()).device
            inputs = {k: v.to(first_device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            with torch.no_grad():
                # Run full model forward just to trigger visual encoder hooks
                model.qwen.model.thinker(**inputs, use_audio_in_video=False)

        except Exception as e:
            print(f"  Frame {fi}: error during forward — {e}", flush=True)
            os.unlink(tmp_path)
            continue

        os.unlink(tmp_path)

        # ── collect captured attention weights ──
        collected = [s[0] for s in all_layer_storage if s]
        if not collected:
            print(f"  Frame {fi}: no attention weights captured. "
                  f"The attention layers may not return weights by default. "
                  f"Run --discover to inspect layer types.", flush=True)
            continue

        print(f"  Frame {fi}: captured {len(collected)} layers, "
              f"shape {collected[0].shape}", flush=True)

        # ── compute patch grid dimensions ──
        # shape: (B, heads, N, N)  where N = 1 (CLS) + h_p * w_p
        N = collected[0].shape[-1]
        n_patches = N - 1  # subtract CLS token
        h_patches = w_patches = int(math.isqrt(n_patches))
        if h_patches * w_patches != n_patches:
            # Non-square — try to find factors
            for h in range(int(n_patches**0.5), 0, -1):
                if n_patches % h == 0:
                    h_patches, w_patches = h, n_patches // h
                    break

        # ── attention rollout or single layer ──
        if args.rollout:
            scores = _rollout([a[0] for a in collected])  # (N,)
        else:
            layer = collected[args.layer_idx]   # (B, heads, N, N)
            scores = layer[0].mean(axis=0)[0]   # mean over heads, CLS row -> (N,)

        # ── overlay and save ──
        vis = _to_heatmap(scores, h_patches, w_patches, frame_rgb)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(frame_rgb);   axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(vis);         axes[1].set_title("Attention"); axes[1].axis("off")
        tag = "rollout" if args.rollout else f"layer{args.layer_idx}"
        ckpt_tag = Path(args.ckpt_path).stem if args.ckpt_path else "untrained"
        fname = out_dir / f"{video_stem}_frame{fi:02d}_{ckpt_tag}_{tag}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  saved {fname}", flush=True)

    for h in hooks:
        h.remove()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
