"""Attention map visualisation for the Qwen3-Omni visual encoder.

The visual encoder uses Qwen3OmniMoeVisionAttention with a fused qkv Linear —
no nn.MultiheadAttention, so standard hooks return nothing. We hook the qkv
Linear output directly and recompute attention weights as softmax(QK^T/sqrt(d)).
There is no CLS token; we average attention over all query positions to get a
per-patch saliency map.

Usage:

  # discover n_heads (run once):
  python -m av_ib.eval.plot_attention_maps --video clip.mp4 --discover

  # plot 6 frames with attention rollout across all 27 layers:
  python -m av_ib.eval.plot_attention_maps \\
      --video clip.mp4 \\
      --out-dir runs/attention_maps \\
      --n-frames 6 --rollout

  # or single last layer:
  python -m av_ib.eval.plot_attention_maps \\
      --video clip.mp4 \\
      --out-dir runs/attention_maps \\
      --n-frames 6 --layer-idx -1

  # with a trained checkpoint:
  python -m av_ib.eval.plot_attention_maps \\
      --video clip.mp4 --out-dir runs/attention_maps \\
      --ckpt-path runs/avqa/v6b_base_final.pt \\
      --n-frames 6 --rollout
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
def _extract_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
    """Uniformly sampled RGB frames as uint8 (H, W, 3)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    idxs = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _infer_n_heads(attn_module: nn.Module) -> int:
    """Infer number of attention heads from the qkv weight shape."""
    qkv: nn.Linear = attn_module.qkv
    # qkv.weight: (3 * d_model, d_model)
    three_d = qkv.weight.shape[0]
    d_model  = qkv.weight.shape[1]
    # Qwen ViT typically uses head_dim=128 or 64
    for head_dim in (128, 64, 32):
        n_heads = d_model // head_dim
        if n_heads * head_dim == d_model and three_d == 3 * d_model:
            return n_heads
    # fallback: assume head_dim=64
    return d_model // 64


def _attn_weights_from_qkv(qkv_out: torch.Tensor, n_heads: int) -> np.ndarray:
    """Compute attention weight matrix from fused QKV output.

    qkv_out: (N, 3*d_model)  — N = number of patch tokens, no batch dim after
             the encoder flattens everything, OR (B, N, 3*d_model).
    Returns: (n_heads, N, N) float32 numpy array.
    """
    if qkv_out.dim() == 2:
        qkv_out = qkv_out.unsqueeze(0)   # (1, N, 3*d)
    B, N, three_d = qkv_out.shape
    d_model = three_d // 3
    head_dim = d_model // n_heads

    q, k, _ = qkv_out.chunk(3, dim=-1)   # each (B, N, d_model)
    # reshape to (B, n_heads, N, head_dim)
    q = q.view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)

    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale    # (B, n_heads, N, N)
    attn = F.softmax(attn, dim=-1)

    # take first batch element, detach
    return attn[0].detach().cpu().float().numpy()   # (n_heads, N, N)


def _rollout(layers: list[np.ndarray], add_residual: bool = True) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020).

    layers: list of (n_heads, N, N) — one per transformer block.
    Returns: (N, N) joint attention matrix. Use mean over rows for saliency.
    """
    R = np.eye(layers[0].shape[-1])
    for a in layers:
        a_mean = a.mean(axis=0)               # (N, N)
        if add_residual:
            a_mean = a_mean + np.eye(a_mean.shape[0])
            a_mean /= a_mean.sum(axis=-1, keepdims=True) + 1e-8
        R = a_mean @ R
    return R   # (N, N)


def _saliency(attn_matrix: np.ndarray) -> np.ndarray:
    """Mean attention over all query positions -> (N,) patch saliency."""
    return attn_matrix.mean(axis=0)


def _to_heatmap(scores: np.ndarray, h_p: int, w_p: int,
                frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Resize patch scores to frame size and blend as a heatmap."""
    import cv2
    grid = scores[:h_p * w_p].reshape(h_p, w_p)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    H, W = frame.shape[:2]
    heat = cv2.resize(grid.astype(np.float32), (W, H),
                      interpolation=cv2.INTER_LINEAR)
    colormap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    return (alpha * colormap + (1 - alpha) * frame).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True)
    ap.add_argument("--out-dir",    default="runs/attention_maps")
    ap.add_argument("--ckpt-path",  default=None)
    ap.add_argument("--variant",    default="b_topk_nofusion")
    ap.add_argument("--n-frames",   type=int, default=6)
    ap.add_argument("--layer-idx",  type=int, default=-1,
                    help="Single layer index to visualise (-1 = last). "
                         "Ignored when --rollout is set.")
    ap.add_argument("--rollout",    action="store_true",
                    help="Aggregate all layers via attention rollout.")
    ap.add_argument("--discover",   action="store_true",
                    help="Print module tree and exit.")
    args = ap.parse_args()

    # ── build model ──────────────────────────────────────────────────────────
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

    if args.discover:
        print("\n=== Visual encoder module tree ===")
        for name, mod in vis_enc.named_modules():
            if name:
                indent = "  " * name.count(".")
                print(f"{indent}{name}: {type(mod).__name__}")
        # also report n_heads
        attn0 = vis_enc.blocks[0].attn
        n_heads = _infer_n_heads(attn0)
        d_model = attn0.qkv.weight.shape[1]
        print(f"\n  d_model={d_model}, n_heads={n_heads}, "
              f"head_dim={d_model // n_heads}")
        print(f"  {len(vis_enc.blocks)} transformer blocks")
        return

    # ── hook qkv linears in every block ──────────────────────────────────────
    attn_blocks = vis_enc.blocks   # ModuleList
    n_heads = _infer_n_heads(attn_blocks[0].attn)
    print(f"n_heads={n_heads}, blocks={len(attn_blocks)}", flush=True)

    # per-block storage: list of lists
    qkv_storage: list[list[torch.Tensor]] = [[] for _ in attn_blocks]

    def _make_qkv_hook(store: list):
        def hook(module, input, output):
            store.append(output.detach())
        return hook

    hooks = []
    for i, blk in enumerate(attn_blocks):
        h = blk.attn.qkv.register_forward_hook(_make_qkv_hook(qkv_storage[i]))
        hooks.append(h)

    # ── extract frames ────────────────────────────────────────────────────────
    frames = _extract_frames(args.video, args.n_frames)
    if not frames:
        raise RuntimeError(f"Could not extract frames from {args.video}")
    print(f"Extracted {len(frames)} frames.", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_stem = Path(args.video).stem
    proc = model.qwen.processor

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tempfile, cv2
    import PIL.Image as PILImage

    ckpt_tag = Path(args.ckpt_path).stem if args.ckpt_path else "untrained"
    tag = "rollout" if args.rollout else f"layer{args.layer_idx}"

    # accumulate (original, heatmap) pairs across all frames
    collected_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for fi, frame_rgb in enumerate(frames):
        for s in qkv_storage:
            s.clear()

        # ── prepare single-image input ────────────────────────────────────
        pil_img = PILImage.fromarray(frame_rgb)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tmp_path = tf.name
        pil_img.save(tmp_path)

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
            _, images_in, _ = process_mm_info(conversation,
                                              use_audio_in_video=False)
            inputs = proc(text=text, images=images_in,
                          return_tensors="pt", padding=True)
            first_dev = next(vis_enc.parameters()).device
            inputs = {k: v.to(first_dev) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            with torch.no_grad():
                model.qwen.model.thinker(**inputs, use_audio_in_video=False)
        except Exception as e:
            print(f"  Frame {fi}: forward error — {e}", flush=True)
            os.unlink(tmp_path)
            continue
        os.unlink(tmp_path)

        # ── compute per-layer attention weights ───────────────────────────
        layer_attns = []
        for store in qkv_storage:
            if not store:
                continue
            w = _attn_weights_from_qkv(store[0], n_heads)
            layer_attns.append(w)

        if not layer_attns:
            print(f"  Frame {fi}: no qkv output captured.", flush=True)
            continue

        N = layer_attns[0].shape[-1]
        h_p = w_p = int(math.isqrt(N))
        if h_p * w_p != N:
            for h in range(int(N ** 0.5), 0, -1):
                if N % h == 0:
                    h_p, w_p = h, N // h
                    break

        print(f"  Frame {fi}: {len(layer_attns)} layers, "
              f"N={N} patches ({h_p}×{w_p})", flush=True)

        if args.rollout:
            scores = _saliency(_rollout(layer_attns))
        else:
            scores = _saliency(layer_attns[args.layer_idx].mean(axis=0))

        vis = _to_heatmap(scores, h_p, w_p, frame_rgb)
        collected_pairs.append((frame_rgb, vis))

    for h in hooks:
        h.remove()

    if not collected_pairs:
        print("No frames processed — nothing to plot.", flush=True)
        return

    # ── single 2×3 figure: top row = originals, bottom row = attention ───────
    n = len(collected_pairs)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))

    for col in range(ncols):
        if col < n:
            orig, attn_vis = collected_pairs[col]
            axes[0, col].imshow(orig)
            axes[0, col].set_title(f"Frame {col}", fontsize=9)
            axes[1, col].imshow(attn_vis)
            axes[1, col].set_title(f"Attention", fontsize=9)
        for row in range(nrows):
            axes[row, col].axis("off")

    # row labels
    axes[0, 0].set_ylabel("Original",  fontsize=10, labelpad=6)
    axes[1, 0].set_ylabel("Attention", fontsize=10, labelpad=6)
    for row in range(nrows):
        axes[row, 0].yaxis.label.set_visible(True)

    fig.suptitle(f"{Path(args.video).stem}  [{ckpt_tag}]  ({tag})", fontsize=11)
    plt.tight_layout()

    fname = out_dir / f"{video_stem}_{ckpt_tag}_{tag}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
