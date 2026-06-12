"""LLM→visual-token attention maps for Qwen3-Omni.

Unlike plot_attention_maps.py (which visualises the frozen visual encoder), this
script captures attention *from inside the LLM* — specifically, attention from
the last question token over all visual-token positions, averaged across the
last N decoder layers. Because LoRA and the bottleneck modify what the LLM
sees, this attention differs across trained checkpoints.

Implementation:
  - Load model with attn_implementation="eager" so output_attentions=True works.
  - Process the video through the normal Qwen3-Omni pipeline (so the same
    visual tokens are produced as during eval).
  - Forward the thinker with output_attentions=True; capture each layer's
    (B, n_heads, L, L) tensor.
  - The last input token (just before "assistant\\n") is the query. Take its
    row of attention to the positions of <|video_pad|> tokens in the sequence.
  - Mean over heads and over the last N layers — earlier layers tend to be
    syntactic / positional, deeper ones encode content.
  - The processor's image_grid_thw = (T, H, W) tells us how many visual tokens
    correspond to each sampled frame, so we reshape (n_visual,) → (T, H, W),
    pick frames uniformly, and overlay the heatmap on the actual RGB frames.

Usage:
    python -m av_ib.eval.plot_llm_attention_maps \\
        --video clip.mp4 \\
        --out-dir runs/attention_maps_llm \\
        --question "What is the person doing?" \\
        --n-frames 6
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch


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


def _to_heatmap(grid: np.ndarray, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    import cv2
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    H, W = frame.shape[:2]
    heat = cv2.resize(grid.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    colormap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    return (alpha * colormap + (1 - alpha) * frame).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True)
    ap.add_argument("--out-dir",    default="runs/attention_maps_llm")
    ap.add_argument("--ckpt-path",  default=None)
    ap.add_argument("--variant",    default="b_topk_nofusion")
    ap.add_argument("--question",   default="What is happening in this video?")
    ap.add_argument("--n-frames",   type=int, default=6,
                    help="Number of frames to display in the figure.")
    ap.add_argument("--last-n-layers", type=int, default=8,
                    help="Average attention over the deepest N decoder layers.")
    ap.add_argument("--mask-sinks", type=int, default=0,
                    help="Zero out top-k saliency outliers per frame. 0 = off.")
    args = ap.parse_args()

    # ── build model with eager attention ─────────────────────────────────────
    print("Loading model with attn_implementation=eager...", flush=True)
    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(
        use_lora=bool(args.ckpt_path),
        variant=args.variant,
        attn_implementation="eager",
    ).eval()
    model.set_sample_noise(False)

    if args.ckpt_path:
        sd = torch.load(args.ckpt_path, map_location="cpu")
        if "trainable_state" in sd:
            sd = sd["trainable_state"]
        own = dict(model.named_parameters())
        n_ok = sum(1 for k, v in sd.items()
                   if k in own and not own[k].data.copy_(v.data) is None)
        print(f"  loaded {n_ok}/{len(sd)} params", flush=True)

    # If trained, install the bottleneck provider so the splicer fires.
    if args.ckpt_path:
        with torch.no_grad():
            provider, _, _, _ = model._make_provider()
        model.qwen._current_provider = provider
    else:
        model.qwen._current_provider = None

    # ── build inputs for the real video + question ───────────────────────────
    print(f"Preparing inputs for {Path(args.video).name}...", flush=True)
    inputs, prompt_len = model.qwen._prep_inputs(args.video, args.question, answer=None)

    tok = model.qwen.tokenizer
    # Qwen3-Omni uses <|video_pad|> for video tokens, <|image_pad|> for images.
    video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    ids = inputs["input_ids"][0]
    vis_mask = (ids == video_pad_id) | (ids == image_pad_id)
    vis_pos = vis_mask.nonzero(as_tuple=True)[0]
    n_vis = vis_pos.numel()
    print(f"  prompt_len={prompt_len}, n_visual_tokens={n_vis}, "
          f"total_len={ids.shape[0]}", flush=True)
    if n_vis == 0:
        raise RuntimeError("No visual tokens found in input_ids — check video processing.")

    # Spatial layout from the processor
    video_grid_thw = inputs.get("video_grid_thw")
    image_grid_thw = inputs.get("image_grid_thw")
    grid_thw = video_grid_thw if video_grid_thw is not None else image_grid_thw
    if grid_thw is None:
        raise RuntimeError("No grid_thw in inputs.")
    # grid_thw shape: (n_media, 3) = [T, H, W] in patch units
    T, H_p, W_p = int(grid_thw[0, 0]), int(grid_thw[0, 1]), int(grid_thw[0, 2])
    # Qwen merges 2x2 spatial patches into one token (spatial_merge_size=2).
    # Number of visual tokens = T * (H/2) * (W/2)
    spatial_merge = 2
    H_t, W_t = H_p // spatial_merge, W_p // spatial_merge
    expected = T * H_t * W_t
    if expected != n_vis:
        # Try alternative merge sizes
        for sm in (1, 2, 4):
            H_t, W_t = H_p // sm, W_p // sm
            if T * H_t * W_t == n_vis:
                spatial_merge = sm
                break
        else:
            print(f"  WARN: T*H/2*W/2={expected} != n_vis={n_vis}; using sqrt fallback")
            side = int(math.isqrt(n_vis // max(T, 1)))
            H_t = W_t = side
    print(f"  grid: T={T}, H_t={H_t}, W_t={W_t}  (merge={spatial_merge})", flush=True)

    # Pick the query position: last token of the prompt (the marker
    # <|im_start|>assistant\\n ends at prompt_len-1).
    q_pos = prompt_len - 1
    print(f"  query position: {q_pos}", flush=True)

    # ── forward thinker with output_attentions ───────────────────────────────
    print("Running thinker forward (output_attentions=True)...", flush=True)
    with torch.no_grad():
        out = model.qwen.model.thinker(
            **inputs,
            output_attentions=True,
            use_audio_in_video=True,
            return_dict=True,
        )

    attns = out.attentions  # tuple len L of (B, H, L, L)
    print(f"  got {len(attns)} layer attentions; "
          f"shape[-1]={attns[-1].shape}", flush=True)

    # ── extract attention from q_pos to visual positions ─────────────────────
    take = attns[-args.last_n_layers:] if args.last_n_layers > 0 else attns
    sal = None
    for a in take:
        # a: (1, H, L, L) — index query row, then gather visual columns
        row = a[0, :, q_pos, :]                 # (H, L)
        row = row[:, vis_pos].mean(dim=0)       # (n_vis,)
        sal = row.float().cpu() if sal is None else sal + row.float().cpu()
    sal = (sal / len(take)).numpy()             # (n_vis,)
    print(f"  saliency: min={sal.min():.3e} max={sal.max():.3e} "
          f"mean={sal.mean():.3e}", flush=True)

    # ── reshape to (T, H_t, W_t) ─────────────────────────────────────────────
    sal_grid = sal[: T * H_t * W_t].reshape(T, H_t, W_t)

    # Optional: suppress per-frame top-k outliers
    if args.mask_sinks > 0:
        for t in range(T):
            flat = sal_grid[t].flatten()
            if args.mask_sinks < flat.size:
                top_idx = np.argpartition(flat, -args.mask_sinks)[-args.mask_sinks:]
                flat[top_idx] = 0.0
            sal_grid[t] = flat.reshape(H_t, W_t)

    # ── extract original frames at matching timesteps ────────────────────────
    rgb_frames = _extract_frames(args.video, args.n_frames)
    if not rgb_frames:
        raise RuntimeError(f"Could not extract frames from {args.video}")

    # Map each displayed frame to a timestep index in sal_grid
    if T == 1:
        # Single image grid: same saliency applied to all rgb_frames
        frame_to_t = [0] * len(rgb_frames)
    else:
        frame_to_t = [round(i * (T - 1) / max(len(rgb_frames) - 1, 1))
                      for i in range(len(rgb_frames))]
    print(f"  frame→t map: {frame_to_t}", flush=True)

    # ── plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rgb_frames)
    fig, axes = plt.subplots(2, n, figsize=(n * 3.2, 6))
    for i, (frame, t_idx) in enumerate(zip(rgb_frames, frame_to_t)):
        vis = _to_heatmap(sal_grid[t_idx], frame)
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f"Frame {i}", fontsize=9)
        axes[1, i].imshow(vis)
        axes[1, i].set_title(f"LLM attn (t={t_idx})", fontsize=9)
        for r in range(2):
            axes[r, i].axis("off")

    ckpt_tag = Path(args.ckpt_path).stem if args.ckpt_path else "untrained"
    fig.suptitle(f"{Path(args.video).stem}  [{ckpt_tag}]  "
                 f"Q: {args.question}", fontsize=10)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{Path(args.video).stem}_{ckpt_tag}_llmattn.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}", flush=True)


if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
