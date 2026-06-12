"""LLM→visual-token attention maps for Qwen3-Omni.

Unlike plot_attention_maps.py (which visualises the frozen visual encoder),
this script captures attention *from inside the LLM thinker* — specifically,
attention from the last query token over all visual-token positions, averaged
across the deepest N decoder layers.  Because LoRA and the bottleneck modify
what the LLM sees, this attention differs across checkpoints.

Implementation avoids output_attentions=True (broken with device_map=auto).
Instead we hook q_proj and k_proj in the last N decoder layers to capture
their outputs, then compute softmax(Q[q_pos] @ K^T / sqrt(d_head)) and index
into the visual-token columns.  No device mismatch: the hooks capture tensors
before they leave each GPU shard.

Usage:
    python -m av_ib.eval.plot_llm_attention_maps \\
        --video clip.mp4 \\
        --out-dir runs/attention_maps_llm \\
        --n-frames 6

    # with a trained checkpoint:
    python -m av_ib.eval.plot_llm_attention_maps \\
        --video clip.mp4 --ckpt-path runs/avqa/v6b_base_final.pt \\
        --out-dir runs/attention_maps_llm --n-frames 6
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
def _extract_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
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


def _attn_row(q: torch.Tensor, k: torch.Tensor,
              q_pos: int, vis_pos: torch.Tensor, n_heads: int) -> np.ndarray:
    """Compute attention weights for token q_pos over vis_pos tokens.

    q, k: (1, L, D) or (L, D) — full sequence Q/K projections.
    Returns (n_vis,) float32 numpy saliency (mean over heads).
    """
    if q.dim() == 2:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
    B, L, D = q.shape
    head_dim = D // n_heads
    # Reshape to (B, n_heads, L, head_dim)
    q = q.view(B, L, n_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(B, L, n_heads, head_dim).permute(0, 2, 1, 3)
    scale = head_dim ** -0.5
    # (B, n_heads, L)
    scores = (q[:, :, q_pos:q_pos+1, :] @ k.transpose(-2, -1)).squeeze(-2) * scale
    attn = F.softmax(scores.float(), dim=-1)   # (B, n_heads, L)
    # gather visual columns then mean over heads and batch
    vis_attn = attn[0, :, vis_pos.to(attn.device)]   # (n_heads, n_vis)
    return vis_attn.mean(dim=0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",         required=True)
    ap.add_argument("--out-dir",       default="runs/attention_maps_llm")
    ap.add_argument("--ckpt-path",     default=None)
    ap.add_argument("--variant",       default="b_topk_nofusion")
    ap.add_argument("--question",      default="What is happening in this video?")
    ap.add_argument("--n-frames",      type=int, default=6)
    ap.add_argument("--last-n-layers", type=int, default=8,
                    help="Average attention over the deepest N decoder layers.")
    ap.add_argument("--mask-sinks",    type=int, default=0,
                    help="Zero out top-k saliency outliers per frame. 0=off.")
    args = ap.parse_args()

    # ── build model (default flash attention is fine — we use hooks, not output_attentions) ─
    print("Loading model...", flush=True)
    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(
        use_lora=bool(args.ckpt_path),
        variant=args.variant,
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

    # ── locate decoder layers ─────────────────────────────────────────────────
    # Path: model.qwen.model.thinker.model.layers  (PEFT wraps .model)
    thinker_model = model.qwen.model.thinker.model
    # PEFT wraps the inner model; .get_base_model() or .base_model.model
    if hasattr(thinker_model, "get_base_model"):
        inner = thinker_model.get_base_model()
    elif hasattr(thinker_model, "base_model"):
        inner = thinker_model.base_model
    else:
        inner = thinker_model
    if hasattr(inner, "model"):
        layers = inner.model.layers
    else:
        layers = inner.layers
    n_layers = len(layers)
    take_idx = list(range(max(0, n_layers - args.last_n_layers), n_layers))
    print(f"  {n_layers} decoder layers, hooking last {len(take_idx)}: {take_idx}", flush=True)

    # Infer n_heads from q_proj of first hooked layer
    q_proj_0 = layers[take_idx[0]].self_attn.q_proj
    D_out = q_proj_0.weight.shape[0]     # output dim of q_proj = n_heads * head_dim
    D_in  = q_proj_0.weight.shape[1]     # hidden_size
    # For GQA models, k_proj may have fewer heads; we only need q_proj head count
    # head_dim is typically 128 in Qwen3
    for hd in (128, 64, 256, 32):
        if D_out % hd == 0:
            n_heads = D_out // hd
            head_dim = hd
            break
    print(f"  q_proj D_out={D_out}, D_in={D_in}, n_heads={n_heads}, head_dim={head_dim}",
          flush=True)

    # ── install hooks on q_proj and k_proj of chosen layers ──────────────────
    q_store: dict[int, torch.Tensor] = {}
    k_store: dict[int, torch.Tensor] = {}
    hooks = []

    def _make_hook(store: dict, idx: int):
        def hook(module, inp, out):
            store[idx] = out.detach()
        return hook

    for i in take_idx:
        sa = layers[i].self_attn
        hooks.append(sa.q_proj.register_forward_hook(_make_hook(q_store, i)))
        hooks.append(sa.k_proj.register_forward_hook(_make_hook(k_store, i)))

    # ── prepare inputs ────────────────────────────────────────────────────────
    print(f"Preparing inputs for {Path(args.video).name}...", flush=True)
    inputs, prompt_len = model.qwen._prep_inputs(args.video, args.question, answer=None)

    tok = model.qwen.tokenizer
    video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    ids = inputs["input_ids"][0]
    vis_mask = (ids == video_pad_id) | (ids == image_pad_id)
    vis_pos  = vis_mask.nonzero(as_tuple=True)[0]   # CPU tensor
    n_vis    = vis_pos.numel()
    print(f"  prompt_len={prompt_len}, n_visual_tokens={n_vis}, "
          f"total_len={ids.shape[0]}", flush=True)
    if n_vis == 0:
        raise RuntimeError("No visual tokens found — check video processing.")

    # Spatial layout
    _vg = inputs.get("video_grid_thw")
    _ig = inputs.get("image_grid_thw")
    grid_thw = _vg if _vg is not None else _ig
    if grid_thw is None:
        raise RuntimeError("No grid_thw in inputs.")
    T, H_p, W_p = int(grid_thw[0, 0]), int(grid_thw[0, 1]), int(grid_thw[0, 2])
    spatial_merge = 2
    H_t, W_t = H_p // spatial_merge, W_p // spatial_merge
    if T * H_t * W_t != n_vis:
        for sm in (1, 2, 4):
            H_t, W_t = H_p // sm, W_p // sm
            if T * H_t * W_t == n_vis:
                spatial_merge = sm
                break
        else:
            side = int(math.isqrt(n_vis // max(T, 1)))
            H_t = W_t = side
    print(f"  grid: T={T}, H_t={H_t}, W_t={W_t}  (merge={spatial_merge})", flush=True)

    q_pos = prompt_len - 1   # last token of the prompt = first assistant token position

    # ── forward via generate (1 token) — proven to work on multi-GPU ─────────
    # Calling .thinker() directly causes device mismatches with device_map=auto.
    # model.generate() goes through the outer Qwen3OmniMoeForConditionalGeneration
    # which has the proper Accelerate AlignDevicesHooks in place.
    print("Running generate (1 token) to trigger prefill forward...", flush=True)
    with torch.no_grad():
        model.qwen.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_audio=False,
            use_audio_in_video=True,
        )

    for h in hooks:
        h.remove()

    print(f"  captured q in {len(q_store)} layers, k in {len(k_store)} layers", flush=True)

    # ── compute per-layer saliency and average ────────────────────────────────
    sal = None
    count = 0
    for i in take_idx:
        if i not in q_store or i not in k_store:
            continue
        row = _attn_row(q_store[i], k_store[i], q_pos, vis_pos, n_heads)
        sal = row if sal is None else sal + row
        count += 1
    if sal is None or count == 0:
        raise RuntimeError("No attention captured — hooks may have missed all layers.")
    sal /= count
    print(f"  saliency ({count} layers): min={sal.min():.3e} max={sal.max():.3e} "
          f"mean={sal.mean():.3e}", flush=True)

    # ── reshape and optional sink suppression ────────────────────────────────
    sal_grid = sal[: T * H_t * W_t].reshape(T, H_t, W_t)
    if args.mask_sinks > 0:
        for t in range(T):
            flat = sal_grid[t].flatten()
            if args.mask_sinks < flat.size:
                top_idx = np.argpartition(flat, -args.mask_sinks)[-args.mask_sinks:]
                flat[top_idx] = 0.0
            sal_grid[t] = flat.reshape(H_t, W_t)

    # ── extract RGB frames and map to timesteps ───────────────────────────────
    rgb_frames = _extract_frames(args.video, args.n_frames)
    if not rgb_frames:
        raise RuntimeError(f"Could not extract frames from {args.video}")
    frame_to_t = ([0] * len(rgb_frames) if T == 1 else
                  [round(i * (T - 1) / max(len(rgb_frames) - 1, 1))
                   for i in range(len(rgb_frames))])
    print(f"  frame→t map: {frame_to_t}", flush=True)

    # ── plot ──────────────────────────────────────────────────────────────────
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
        axes[1, i].set_title(f"attn t={t_idx}", fontsize=9)
        for r in range(2):
            axes[r, i].axis("off")

    ckpt_tag = Path(args.ckpt_path).stem if args.ckpt_path else "untrained"
    fig.suptitle(f"{Path(args.video).stem}  [{ckpt_tag}]  Q: {args.question}",
                 fontsize=10)
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
