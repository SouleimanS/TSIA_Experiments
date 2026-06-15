"""Token analysis: video/audio embedding norms, cosine similarity, and LLM attention
distribution with vs without the VIB.

Answers: what happens to video/audio token weights (norms) and LLM attention when
the VIB is active vs not?

Usage:
    python -m av_ib.token_analysis.analyse \\
        --video clip1.mp4 --video clip2.mp4 \\
        --ckpt-path runs/avqa/v6b_base_final.pt \\
        --variant b_topk_nofusion \\
        --out-dir results/token_analysis

    # or with a directory of clips:
    python -m av_ib.token_analysis.analyse \\
        --video-dir /data/avhbench \\
        --n-clips 10 \\
        --ckpt-path runs/avqa/v6b_base_final.pt \\
        --out-dir results/token_analysis
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Attention helpers
# ─────────────────────────────────────────────────────────────────────────────

def _attn_row_full(q: torch.Tensor, k: torch.Tensor,
                   q_pos: int, n_heads: int) -> np.ndarray:
    """Compute attention weights for token q_pos over ALL token positions.

    q: (1, L, n_heads*head_dim) or (L, n_heads*head_dim)
    k: (1, L, n_kv_heads*head_dim) — GQA: n_kv_heads may differ from n_heads.
    Returns (L,) float32 numpy array (mean over query heads).
    """
    if q.dim() == 2:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
    B, L, Dq = q.shape
    Dk = k.shape[-1]
    head_dim = Dq // n_heads
    n_kv_heads = Dk // head_dim
    groups = n_heads // n_kv_heads   # queries per KV head

    q = q.view(B, L, n_heads, head_dim).permute(0, 2, 1, 3)        # (B, H, L, d)
    k = k.view(B, L, n_kv_heads, head_dim).permute(0, 2, 1, 3)     # (B, Hkv, L, d)
    k = k.repeat_interleave(groups, dim=1)                           # (B, H, L, d)

    scale = head_dim ** -0.5
    # scores: (B, H, 1, L) — attention from q_pos over all positions
    scores = (q[:, :, q_pos:q_pos+1, :] @ k.transpose(-2, -1)).squeeze(-2) * scale
    attn = F.softmax(scores.float(), dim=-1)   # (B, H, L)
    # mean over heads, squeeze batch
    return attn[0].mean(dim=0).cpu().numpy()   # (L,)


# ─────────────────────────────────────────────────────────────────────────────
# Mask extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _mask_to_pos(mask: torch.Tensor) -> torch.Tensor:
    """Convert a 2D or 3D boolean mask to a 1D position tensor (first batch item)."""
    if mask.dim() == 3:
        m = mask.any(dim=-1)[0]   # (L,)
    else:
        m = mask[0]               # (L,)
    return m.nonzero(as_tuple=True)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition forward: embeddings + full-sequence attention
# ─────────────────────────────────────────────────────────────────────────────

def _run_condition(model, inputs, take_idx, layers, n_heads, q_pos,
                   vid_pos: torch.Tensor, aud_pos: torch.Tensor):
    """Run one forward pass and return embedding and attention metrics.

    Returns a dict with:
      vid_embeds: (N_v, D) float32 tensor (cpu)
      aud_embeds: (N_a, D) float32 tensor (cpu)
      attn_full:  (L,)    float32 numpy (mean over heads and last-N layers)
    """
    q_store: dict[int, torch.Tensor] = {}
    k_store: dict[int, torch.Tensor] = {}
    embeds_store: dict[str, torch.Tensor] = {}
    hooks = []

    def _mk_attn(store, idx):
        def hook(module, inp, out):
            store[idx] = out.detach()
        return hook

    def _mk_embed(store):
        def hook(module, args):
            # pre-hook: args is a tuple; first element is hidden states entering layernorm
            x = args[0].detach()   # (B, L, D)
            store["embeds"] = x[0].float().cpu()   # (L, D)
        return hook

    # Hook embeddings at layer[0].input_layernorm
    first_layer = layers[take_idx[0] - (take_idx[0] - 0)]  # layer 0
    # We want the actual first layer (index 0), not first of take_idx
    # Recompute: layers is the full layers list; layer 0 is layers[0]
    hooks.append(layers[0].input_layernorm.register_forward_pre_hook(_mk_embed(embeds_store)))

    # Hook q/k projections in the last-N layers
    for i in take_idx:
        sa = layers[i].self_attn
        hooks.append(sa.q_proj.register_forward_hook(_mk_attn(q_store, i)))
        hooks.append(sa.k_proj.register_forward_hook(_mk_attn(k_store, i)))

    try:
        with torch.no_grad():
            model.qwen.model.thinker(**inputs, use_audio_in_video=True)
    finally:
        for h in hooks:
            h.remove()

    # ── embeddings ────────────────────────────────────────────────────────────
    embeds = embeds_store.get("embeds")   # (L, D) or None
    if embeds is None:
        raise RuntimeError("Embedding hook did not fire.")

    vid_embeds = embeds[vid_pos.cpu()]    # (N_v, D)
    aud_embeds = embeds[aud_pos.cpu()]    # (N_a, D)

    # ── attention (mean over last-N layers, full sequence) ────────────────────
    attn_sum = None
    count = 0
    for i in take_idx:
        if i in q_store and i in k_store:
            row = _attn_row_full(q_store[i], k_store[i], q_pos, n_heads)
            attn_sum = row if attn_sum is None else attn_sum + row
            count += 1
    if attn_sum is None:
        raise RuntimeError("No attention captured.")
    attn_full = attn_sum / count   # (L,)

    # ── key norms at vis/aud positions (mean over last-N layers, all heads) ──
    # k_store[i] shape: (B, L, D_k); reshape to (B, n_kv_heads, L, head_dim)
    k_norm_vid_sum = 0.0
    k_norm_aud_sum = 0.0
    k_norm_txt_sum = 0.0
    kn_count = 0
    vid_cpu = vid_pos.cpu()
    aud_cpu = aud_pos.cpu()
    L_seq = embeds.shape[0]
    txt_mask = torch.ones(L_seq, dtype=torch.bool)
    txt_mask[vid_cpu] = False
    txt_mask[aud_cpu] = False
    for i in take_idx:
        if i in k_store:
            k = k_store[i][0].float()   # (L, D_k)
            k_norm_vid_sum += float(k[vid_cpu].norm(dim=-1).mean())
            k_norm_aud_sum += float(k[aud_cpu].norm(dim=-1).mean())
            k_norm_txt_sum += float(k[txt_mask].norm(dim=-1).mean())
            kn_count += 1
    k_norm_vid = k_norm_vid_sum / max(kn_count, 1)
    k_norm_aud = k_norm_aud_sum / max(kn_count, 1)
    k_norm_txt = k_norm_txt_sum / max(kn_count, 1)

    return {
        "vid_embeds": vid_embeds,
        "aud_embeds": aud_embeds,
        "attn_full": attn_full,
        "k_norm_vid": k_norm_vid,
        "k_norm_aud": k_norm_aud,
        "k_norm_txt": k_norm_txt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_clip(model, video_path: str, question: str,
                  take_idx: list[int], layers, n_heads: int,
                  out_dir: Path) -> dict:
    """Run both conditions on one clip and return metrics dict."""
    stem = Path(video_path).stem
    print(f"\n[{stem}] Preparing inputs...", flush=True)

    inputs, prompt_len = model.qwen._prep_inputs(video_path, question, answer=None)
    q_pos = prompt_len - 1
    L = inputs["input_ids"].shape[1]
    print(f"  prompt_len={prompt_len}, total_len={L}", flush=True)

    # ── get video/audio token positions via splicer mask ─────────────────────
    splicer = model.qwen.splicer
    orig_clear = splicer._clear_per_call_state
    splicer._clear_per_call_state = lambda: None   # prevent mask clearing

    # Run a lightweight forward (raw condition) to capture masks
    model.qwen._current_provider = None
    with torch.no_grad():
        model.qwen.model.thinker(**inputs, use_audio_in_video=True)

    vid_mask = splicer._video_mask
    aud_mask = splicer._audio_mask

    if vid_mask is None or aud_mask is None:
        # Restore and clean up before raising
        splicer._clear_per_call_state = orig_clear
        splicer._clear_per_call_state()
        raise RuntimeError(f"[{stem}] Splicer masks not captured — likely not an AV clip.")

    vid_pos = _mask_to_pos(vid_mask)   # (N_v,)
    aud_pos = _mask_to_pos(aud_mask)   # (N_a,)

    # Restore and clean up
    splicer._clear_per_call_state = orig_clear
    splicer._clear_per_call_state()

    n_video = int(vid_pos.numel())
    n_audio = int(aud_pos.numel())
    print(f"  n_video={n_video}, n_audio={n_audio}", flush=True)

    if n_video == 0 or n_audio == 0:
        raise RuntimeError(f"[{stem}] Zero-length modality: n_video={n_video}, n_audio={n_audio}")

    # ── two conditions ────────────────────────────────────────────────────────
    results = {}
    for mode in ("raw", "vib"):
        if mode == "raw":
            model.qwen._current_provider = None
        else:
            provider, _, _, _ = model._make_provider()
            model.qwen._current_provider = provider

        print(f"  [{stem}] Running forward [{mode}]...", flush=True)
        try:
            res = _run_condition(model, inputs, take_idx, layers, n_heads,
                                 q_pos, vid_pos, aud_pos)
        finally:
            model.qwen._current_provider = None

        results[mode] = res

    # ── compute metrics ───────────────────────────────────────────────────────
    def _norm_mean(t: torch.Tensor) -> float:
        return float(t.norm(dim=-1).mean().item())

    def _cos_sim_mean(a: torch.Tensor, b: torch.Tensor) -> float:
        # per-token cosine similarity, then mean
        sims = F.cosine_similarity(a.float(), b.float(), dim=-1)
        return float(sims.mean().item())

    def _attn_fracs(attn_full: np.ndarray, vid_pos, aud_pos, L):
        total = float(attn_full.sum())
        if total < 1e-12:
            return 0.0, 0.0, 0.0
        vid_mass = float(attn_full[vid_pos.cpu().numpy()].sum())
        aud_mass = float(attn_full[aud_pos.cpu().numpy()].sum())
        txt_mass = max(0.0, total - vid_mass - aud_mass)
        return vid_mass / total, aud_mass / total, txt_mass / total

    vid_norm_raw = _norm_mean(results["raw"]["vid_embeds"])
    vid_norm_vib = _norm_mean(results["vib"]["vid_embeds"])
    aud_norm_raw = _norm_mean(results["raw"]["aud_embeds"])
    aud_norm_vib = _norm_mean(results["vib"]["aud_embeds"])

    vid_cos = _cos_sim_mean(results["raw"]["vid_embeds"], results["vib"]["vid_embeds"])
    aud_cos = _cos_sim_mean(results["raw"]["aud_embeds"], results["vib"]["aud_embeds"])

    av_raw, aa_raw, at_raw = _attn_fracs(results["raw"]["attn_full"], vid_pos, aud_pos, L)
    av_vib, aa_vib, at_vib = _attn_fracs(results["vib"]["attn_full"], vid_pos, aud_pos, L)

    # Per-token attention: mass / n_tokens — removes token-count effect
    n_txt = max(1, L - n_video - n_audio)
    pt_vid_raw = av_raw / max(n_video, 1)
    pt_aud_raw = aa_raw / max(n_audio, 1)
    pt_txt_raw = at_raw / max(n_txt, 1)
    pt_vid_vib = av_vib / max(n_video, 1)
    pt_aud_vib = aa_vib / max(n_audio, 1)
    pt_txt_vib = at_vib / max(n_txt, 1)

    # Uniform baseline: 1/L per token regardless of modality
    uniform = 1.0 / max(L, 1)

    metrics = {
        "vid_norm_raw": vid_norm_raw,
        "vid_norm_vib": vid_norm_vib,
        "aud_norm_raw": aud_norm_raw,
        "aud_norm_vib": aud_norm_vib,
        "vid_cos_sim": vid_cos,
        "aud_cos_sim": aud_cos,
        "attn_video_raw": av_raw,
        "attn_video_vib": av_vib,
        "attn_audio_raw": aa_raw,
        "attn_audio_vib": aa_vib,
        "attn_text_raw":  at_raw,
        "attn_text_vib":  at_vib,
        # per-token attention (normalized by count)
        "pt_vid_raw": pt_vid_raw,
        "pt_aud_raw": pt_aud_raw,
        "pt_txt_raw": pt_txt_raw,
        "pt_vid_vib": pt_vid_vib,
        "pt_aud_vib": pt_aud_vib,
        "pt_txt_vib": pt_txt_vib,
        "uniform_per_token": uniform,
        # key norms: which modality has larger keys (attracts more softmax mass)
        "k_norm_vid_raw": results["raw"]["k_norm_vid"],
        "k_norm_aud_raw": results["raw"]["k_norm_aud"],
        "k_norm_txt_raw": results["raw"]["k_norm_txt"],
        "k_norm_vid_vib": results["vib"]["k_norm_vid"],
        "k_norm_aud_vib": results["vib"]["k_norm_aud"],
        "k_norm_txt_vib": results["vib"]["k_norm_txt"],
        "n_video": n_video,
        "n_audio": n_audio,
        "n_text": n_txt,
        "n_total": L,
        "video": stem,
    }

    print(f"  [{stem}] per-token attn  raw: vid={pt_vid_raw*1e4:.2f}e-4  aud={pt_aud_raw*1e4:.2f}e-4  txt={pt_txt_raw*1e4:.2f}e-4  (uniform={uniform*1e4:.2f}e-4)", flush=True)
    print(f"  [{stem}] key norms  raw: vid={results['raw']['k_norm_vid']:.3f}  aud={results['raw']['k_norm_aud']:.3f}  txt={results['raw']['k_norm_txt']:.3f}", flush=True)
    print(f"  [{stem}] key norms  vib: vid={results['vib']['k_norm_vid']:.3f}  aud={results['vib']['k_norm_aud']:.3f}  txt={results['vib']['k_norm_txt']:.3f}", flush=True)

    # ── per-clip JSON ─────────────────────────────────────────────────────────
    (out_dir / f"{stem}_token_analysis.json").write_text(json.dumps(metrics, indent=2))

    # ── per-clip figure ───────────────────────────────────────────────────────
    _plot_clip(stem, results, metrics, vid_pos, aud_pos, L, n_heads,
               take_idx, out_dir)

    print(f"  [{stem}] vid_norm raw={vid_norm_raw:.3f} vib={vid_norm_vib:.3f} "
          f"| aud_norm raw={aud_norm_raw:.3f} vib={aud_norm_vib:.3f}", flush=True)
    print(f"  [{stem}] attn_video raw={av_raw:.4f} vib={av_vib:.4f} "
          f"| attn_audio raw={aa_raw:.4f} vib={aa_vib:.4f}", flush=True)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip figure
# ─────────────────────────────────────────────────────────────────────────────

def _build_attn_heatmap(attn_full: np.ndarray, take_idx: list[int], q_store, k_store,
                         q_pos: int, n_heads: int) -> np.ndarray:
    """Build (n_layers, L) attention heatmap for the last-N layers."""
    rows = []
    for i in take_idx:
        if i in q_store and i in k_store:
            row = _attn_row_full(q_store[i], k_store[i], q_pos, n_heads)
            rows.append(row)
    if not rows:
        return np.array([[]])
    return np.stack(rows, axis=0)   # (n_layers, L)


def _plot_clip(stem: str, results: dict, metrics: dict,
               vid_pos: torch.Tensor, aud_pos: torch.Tensor,
               L: int, n_heads: int, take_idx: list[int], out_dir: Path):
    """Clean 4-panel figure:
       top (wide): attention vs token index, raw vs VIB, modality regions shaded
       bottom-left:   attention PER TOKEN (count-normalized) by modality
       bottom-mid:    total attention share by modality
       bottom-right:  key norms by modality (the mechanism)
    """
    attn_raw = results["raw"]["attn_full"]   # (L,)
    attn_vib = results["vib"]["attn_full"]   # (L,)
    vp = vid_pos.cpu().numpy()
    ap = aud_pos.cpu().numpy()

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.3)
    RAW_C, VIB_C = "#1f77b4", "#e67814"

    # ── top: attention across the sequence ────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    tok = np.arange(L)
    ax.plot(tok, attn_raw, color=RAW_C, lw=0.6, alpha=0.8, label="raw")
    ax.plot(tok, attn_vib, color=VIB_C, lw=0.6, alpha=0.8, label="VIB")
    if len(vp) > 0:
        ax.axvspan(vp[0], vp[-1], color="#1f77b4", alpha=0.07)
        ax.text((vp[0]+vp[-1])/2, ax.get_ylim()[1]*0.92, "VIDEO",
                ha="center", fontsize=9, color="#1f77b4", weight="bold")
    if len(ap) > 0:
        ax.axvspan(ap[0], ap[-1], color="#2ca02c", alpha=0.10)
        ax.text((ap[0]+ap[-1])/2, ax.get_ylim()[1]*0.92, "AUDIO",
                ha="center", fontsize=9, color="#2ca02c", weight="bold")
    ax.set_xlabel("Token index"); ax.set_ylabel("Attention from answer token")
    ax.set_title("Where attention lands across the input sequence", fontsize=12, weight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.margins(x=0)

    mods = ["Video", "Audio", "Text"]
    x = np.arange(3); w = 0.38

    # ── bottom-left: per-token attention (the headline) ───────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    pt_raw = [metrics["pt_vid_raw"], metrics["pt_aud_raw"], metrics["pt_txt_raw"]]
    pt_vib = [metrics["pt_vid_vib"], metrics["pt_aud_vib"], metrics["pt_txt_vib"]]
    ax1.bar(x - w/2, np.array(pt_raw)*1e4, w, label="raw", color=RAW_C)
    ax1.bar(x + w/2, np.array(pt_vib)*1e4, w, label="VIB", color=VIB_C)
    ax1.axhline(metrics["uniform_per_token"]*1e4, color="gray", ls="--", lw=1,
                label="uniform")
    ax1.set_xticks(x); ax1.set_xticklabels(mods)
    ax1.set_ylabel(r"Attention per token ($\times10^{-4}$)")
    ax1.set_title("Attention PER TOKEN\n(count-normalized)", fontsize=11, weight="bold")
    ax1.legend(fontsize=8)

    # ── bottom-mid: total attention share ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    tot_raw = [metrics["attn_video_raw"], metrics["attn_audio_raw"], metrics["attn_text_raw"]]
    tot_vib = [metrics["attn_video_vib"], metrics["attn_audio_vib"], metrics["attn_text_vib"]]
    ax2.bar(x - w/2, tot_raw, w, label="raw", color=RAW_C)
    ax2.bar(x + w/2, tot_vib, w, label="VIB", color=VIB_C)
    ax2.set_xticks(x); ax2.set_xticklabels(mods)
    ax2.set_ylabel("Fraction of total attention")
    ax2.set_title(f"Total attention share\n(n_vid={metrics['n_video']}, n_aud={metrics['n_audio']})",
                  fontsize=11, weight="bold")
    ax2.legend(fontsize=8)

    # ── bottom-right: key norms (mechanism) ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    kn_raw = [metrics["k_norm_vid_raw"], metrics["k_norm_aud_raw"], metrics["k_norm_txt_raw"]]
    kn_vib = [metrics["k_norm_vid_vib"], metrics["k_norm_aud_vib"], metrics["k_norm_txt_vib"]]
    ax3.bar(x - w/2, kn_raw, w, label="raw", color=RAW_C)
    ax3.bar(x + w/2, kn_vib, w, label="VIB", color=VIB_C)
    ax3.set_xticks(x); ax3.set_xticklabels(mods)
    ax3.set_ylabel("Mean key norm")
    ax3.set_title("Key norms\n(bigger key = more attention)", fontsize=11, weight="bold")
    ax3.legend(fontsize=8)

    fig.suptitle(
        f"{stem}   |   audio cos-sim={metrics['aud_cos_sim']:.2f} "
        f"(VIB rotates audio)   |   audio key {metrics['k_norm_aud_raw']:.1f}"
        f"→{metrics['k_norm_aud_vib']:.1f}",
        fontsize=13, weight="bold", y=0.99)

    plt.tight_layout()
    fname = out_dir / f"{stem}_token_analysis.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved per-clip figure: {fname}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Summary figure
# ─────────────────────────────────────────────────────────────────────────────

def _plot_summary(all_metrics: list[dict], out_dir: Path, datasets: dict[str, str]):
    """4-subplot summary figure across all clips."""
    if not all_metrics:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Token analysis summary — video/audio norm, cosine sim, attention mass "
                 "(raw vs VIB)", fontsize=13)

    # ── Subplot 1: Attention mass by modality ─────────────────────────────────
    ax = axes[0, 0]
    mean_raw = {
        "Video": np.mean([m["attn_video_raw"] for m in all_metrics]),
        "Audio": np.mean([m["attn_audio_raw"] for m in all_metrics]),
        "Text":  np.mean([m["attn_text_raw"]  for m in all_metrics]),
    }
    mean_vib = {
        "Video": np.mean([m["attn_video_vib"] for m in all_metrics]),
        "Audio": np.mean([m["attn_audio_vib"] for m in all_metrics]),
        "Text":  np.mean([m["attn_text_vib"]  for m in all_metrics]),
    }
    modalities = list(mean_raw.keys())
    x = np.arange(len(modalities))
    w = 0.35
    ax.bar(x - w/2, [mean_raw[k] for k in modalities], w, label="raw", color="steelblue")
    ax.bar(x + w/2, [mean_vib[k] for k in modalities], w, label="VIB", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylabel("Fraction of total attention")
    ax.set_title("LLM attention distribution by modality")
    ax.legend()

    # ── Subplot 2: Norm ratio ─────────────────────────────────────────────────
    ax = axes[0, 1]
    vid_ratios = [m["vid_norm_vib"] / (m["vid_norm_raw"] + 1e-9) for m in all_metrics]
    aud_ratios = [m["aud_norm_vib"] / (m["aud_norm_raw"] + 1e-9) for m in all_metrics]
    labels = ["Video", "Audio"]
    means = [np.mean(vid_ratios), np.mean(aud_ratios)]
    errs  = [np.std(vid_ratios),  np.std(aud_ratios)]
    ax.bar(labels, means, yerr=errs, color=["steelblue", "darkorange"],
           capsize=5, alpha=0.8)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("norm_vib / norm_raw")
    ax.set_title("VIB effect on token L2 norm")

    # ── Subplot 3: Cosine similarity ──────────────────────────────────────────
    ax = axes[1, 0]
    vid_cos = [m["vid_cos_sim"] for m in all_metrics]
    aud_cos = [m["aud_cos_sim"] for m in all_metrics]
    labels = ["Video", "Audio"]
    means = [np.mean(vid_cos), np.mean(aud_cos)]
    errs  = [np.std(vid_cos),  np.std(aud_cos)]
    ax.bar(labels, means, yerr=errs, color=["steelblue", "darkorange"],
           capsize=5, alpha=0.8)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Similarity of VIB vs raw token embeddings")

    # ── Subplot 4: Per-clip scatter ───────────────────────────────────────────
    ax = axes[1, 1]
    dataset_colors = {"AVHBench": "steelblue", "AVQA": "darkorange"}
    default_color = "gray"
    for m in all_metrics:
        dv = m["attn_video_vib"] - m["attn_video_raw"]
        da = m["attn_audio_vib"] - m["attn_audio_raw"]
        ds = datasets.get(m["video"], "other")
        col = dataset_colors.get(ds, default_color)
        ax.scatter(dv, da, color=col, s=50, alpha=0.8, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Δ attn_video (VIB − raw)")
    ax.set_ylabel("Δ attn_audio (VIB − raw)")
    ax.set_title("Per-clip attention shift (video vs audio)")
    # Legend
    for ds_name, col in dataset_colors.items():
        ax.scatter([], [], color=col, label=ds_name, s=50)
    ax.scatter([], [], color=default_color, label="other", s=50)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fname = out_dir / "token_analysis_summary.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"\nSaved summary figure: {fname}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(ckpt_path: str | None, variant: str):
    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(
        use_lora=bool(ckpt_path),
        variant=variant,
        device_map="cuda:0",
    ).eval()
    model.set_sample_noise(False)   # deterministic z = mu

    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        if "trainable_state" in sd:
            sd = sd["trainable_state"]
        own = dict(model.named_parameters())
        n_ok = sum(1 for k, v in sd.items()
                   if k in own and not own[k].data.copy_(v.data) is None)
        print(f"  Loaded {n_ok}/{len(sd)} params from {ckpt_path}", flush=True)

    return model


def _get_layers(model):
    """Locate decoder layers list from the thinker."""
    thinker_model = model.qwen.model.thinker.model
    if hasattr(thinker_model, "get_base_model"):
        inner = thinker_model.get_base_model()
    elif hasattr(thinker_model, "base_model"):
        inner = thinker_model.base_model
    else:
        inner = thinker_model
    if hasattr(inner, "model"):
        return inner.model.layers
    return inner.layers


def _infer_n_heads(layers, take_idx: list[int]) -> int:
    q_proj_0 = layers[take_idx[0]].self_attn.q_proj
    D_out = q_proj_0.weight.shape[0]
    for hd in (128, 64, 256, 32):
        if D_out % hd == 0:
            return D_out // hd
    raise ValueError(f"Cannot infer n_heads from D_out={D_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Analyse video/audio token norms and LLM attention with vs without VIB."
    )
    ap.add_argument("--video", action="append", default=[], metavar="PATH",
                    help="Video file path (can be repeated).")
    ap.add_argument("--video-dir", default=None, metavar="DIR",
                    help="Directory of *.mp4 clips (sorted, first --n-clips taken).")
    ap.add_argument("--n-clips", type=int, default=10,
                    help="Max clips to process from --video-dir.")
    ap.add_argument("--ckpt-path", default=None,
                    help="Trained checkpoint (.pt). If omitted, uses un-trained model.")
    ap.add_argument("--variant", default="b_topk_nofusion",
                    help="Model variant string.")
    ap.add_argument("--question", default="What is happening in this video?")
    ap.add_argument("--out-dir", default="results/token_analysis")
    ap.add_argument("--last-n-layers", type=int, default=8,
                    help="Number of final decoder layers to hook for attention.")
    ap.add_argument("--dataset", action="append", nargs=2, metavar=("STEM", "NAME"),
                    default=[],
                    help="Assign clip stem to a dataset name for the scatter plot. "
                         "E.g. --dataset myclip AVHBench")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect video paths
    video_paths: list[str] = list(args.video)
    if args.video_dir:
        vd = Path(args.video_dir)
        found = sorted(vd.glob("*.mp4"))[:args.n_clips]
        video_paths.extend(str(p) for p in found)

    if not video_paths:
        ap.error("Provide at least one --video PATH or --video-dir DIR.")

    # Dataset mapping (stem -> dataset name)
    datasets: dict[str, str] = dict(args.dataset)

    print("Loading model...", flush=True)
    model = _load_model(args.ckpt_path, args.variant)
    layers = _get_layers(model)
    n_layers = len(layers)
    take_idx = list(range(max(0, n_layers - args.last_n_layers), n_layers))
    n_heads = _infer_n_heads(layers, take_idx)
    print(f"  {n_layers} decoder layers, hooking {take_idx}, n_heads={n_heads}", flush=True)

    all_metrics: list[dict] = []
    failed: list[str] = []

    for vp in video_paths:
        try:
            m = _analyse_clip(model, vp, args.question, take_idx, layers, n_heads, out_dir)
            all_metrics.append(m)
        except Exception as exc:
            print(f"  [ERROR] {Path(vp).stem}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            failed.append(vp)

    if not all_metrics:
        print("No clips succeeded — no summary to write.", flush=True)
        return

    # Summary JSON
    summary = {
        "n_clips": len(all_metrics),
        "failed": failed,
        "mean_vid_norm_ratio": float(np.mean(
            [m["vid_norm_vib"] / (m["vid_norm_raw"] + 1e-9) for m in all_metrics])),
        "mean_aud_norm_ratio": float(np.mean(
            [m["aud_norm_vib"] / (m["aud_norm_raw"] + 1e-9) for m in all_metrics])),
        "mean_vid_cos_sim": float(np.mean([m["vid_cos_sim"] for m in all_metrics])),
        "mean_aud_cos_sim": float(np.mean([m["aud_cos_sim"] for m in all_metrics])),
        "mean_attn_video_raw": float(np.mean([m["attn_video_raw"] for m in all_metrics])),
        "mean_attn_video_vib": float(np.mean([m["attn_video_vib"] for m in all_metrics])),
        "mean_attn_audio_raw": float(np.mean([m["attn_audio_raw"] for m in all_metrics])),
        "mean_attn_audio_vib": float(np.mean([m["attn_audio_vib"] for m in all_metrics])),
        "mean_attn_text_raw":  float(np.mean([m["attn_text_raw"]  for m in all_metrics])),
        "mean_attn_text_vib":  float(np.mean([m["attn_text_vib"]  for m in all_metrics])),
        "clips": all_metrics,
    }
    summary_json = out_dir / "token_analysis_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary JSON: {summary_json}", flush=True)

    _plot_summary(all_metrics, out_dir, datasets)

    print("\nDone.", flush=True)
    print(f"  mean_vid_norm_ratio = {summary['mean_vid_norm_ratio']:.4f}", flush=True)
    print(f"  mean_aud_norm_ratio = {summary['mean_aud_norm_ratio']:.4f}", flush=True)
    print(f"  mean_vid_cos_sim    = {summary['mean_vid_cos_sim']:.4f}", flush=True)
    print(f"  mean_aud_cos_sim    = {summary['mean_aud_cos_sim']:.4f}", flush=True)
    print(f"  Δ attn_video (mean) = "
          f"{summary['mean_attn_video_vib'] - summary['mean_attn_video_raw']:+.4f}", flush=True)
    print(f"  Δ attn_audio (mean) = "
          f"{summary['mean_attn_audio_vib'] - summary['mean_attn_audio_raw']:+.4f}", flush=True)


if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
