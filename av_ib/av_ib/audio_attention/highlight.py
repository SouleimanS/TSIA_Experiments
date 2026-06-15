"""Highlight which parts of the audio receive the most attention.

For each clip we compute gradient x input saliency of the model's own answer
token w.r.t. every input token (clean, causal — no attention sinks). We then
take the saliency at the AUDIO token positions, map each audio token back to
its time in the clip, and overlay that attention curve on the waveform. The
top-attended audio segments are marked.

Uses the video-only model (b_video_only): the bottleneck is on video, audio
passes through untouched, so this shows the audio attention of the actual model.

Usage:
    python -m av_ib.audio_attention.highlight \\
        --video clip.mp4 --ckpt-path runs/avqa/video_bottleneck_sft_final.pt \\
        --variant b_video_only --out-dir results/audio_attention
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
def _audio_positions(model, inputs) -> torch.Tensor:
    """Audio token positions in the sequence, via the splicer's audio mask."""
    splicer = model.qwen.splicer
    orig_clear = splicer._clear_per_call_state
    splicer._clear_per_call_state = lambda: None      # keep masks after forward
    model.qwen._current_provider = None
    try:
        with torch.no_grad():
            model.qwen.model.thinker(**inputs, use_audio_in_video=True)
        aud_mask = splicer._audio_mask
    finally:
        splicer._clear_per_call_state = orig_clear
        splicer._clear_per_call_state()
    if aud_mask is None:
        raise RuntimeError("No audio mask captured (clip has no audio tokens?).")
    m = aud_mask.any(dim=-1)[0] if aud_mask.dim() == 3 else aud_mask[0]
    return m.nonzero(as_tuple=True)[0]


def _grad_saliency(model, inputs, layers, q_pos) -> np.ndarray:
    """Gradient x input saliency over the full sequence (L,)."""
    box: dict[str, torch.Tensor] = {}

    def pre_hook(module, args):
        x = args[0].detach().requires_grad_(True)
        box["x"] = x
        return (x,) + args[1:]

    h = layers[0].register_forward_pre_hook(pre_hook)
    try:
        out = model.qwen.model.thinker(**inputs, use_audio_in_video=True,
                                       use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out[0]
        row = logits[0, q_pos].float()
        target = int(row.argmax().item())
        loss = -torch.log_softmax(row, dim=-1)[target]
        grad = torch.autograd.grad(loss, box["x"])[0]
    finally:
        h.remove()
    sal = torch.relu((grad[0] * box["x"].detach()[0]).sum(dim=-1))
    return sal.float().cpu().numpy()


def _load_waveform(video_path: str, sr: int = 16000):
    import librosa
    y, _ = librosa.load(video_path, sr=sr, mono=True)
    return y, sr


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",     required=True)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--variant",   default="b_video_only")
    ap.add_argument("--question",  default="What is happening in this video?")
    ap.add_argument("--out-dir",   default="results/audio_attention")
    ap.add_argument("--top-k",     type=int, default=3,
                    help="Number of top audio segments to mark.")
    args = ap.parse_args()

    print("Loading model onto cuda:0...", flush=True)
    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(use_lora=True, variant=args.variant,
                      device_map="cuda:0").eval()
    model.set_sample_noise(False)
    try:
        model.qwen.model.thinker.gradient_checkpointing_enable()
    except Exception:
        pass

    sd = torch.load(args.ckpt_path, map_location="cpu")
    if "trainable_state" in sd:
        sd = sd["trainable_state"]
    own = dict(model.named_parameters())
    n_ok = sum(1 for k, v in sd.items()
               if k in own and not own[k].data.copy_(v.data) is None)
    print(f"  loaded {n_ok}/{len(sd)} params", flush=True)

    # decoder layers
    thinker_model = model.qwen.model.thinker.model
    inner = (thinker_model.get_base_model() if hasattr(thinker_model, "get_base_model")
             else getattr(thinker_model, "base_model", thinker_model))
    layers = inner.model.layers if hasattr(inner, "model") else inner.layers

    stem = Path(args.video).stem
    print(f"Preparing inputs for {stem}...", flush=True)
    inputs, prompt_len = model.qwen._prep_inputs(args.video, args.question, answer=None)
    q_pos = prompt_len - 1

    aud_pos = _audio_positions(model, inputs)
    n_aud = int(aud_pos.numel())
    if n_aud == 0:
        raise RuntimeError("No audio tokens.")
    print(f"  n_audio tokens = {n_aud}", flush=True)

    model.qwen._current_provider = None  # video-only ckpt; audio is identity anyway
    sal_full = _grad_saliency(model, inputs, layers, q_pos)
    aud_sal = sal_full[aud_pos.cpu().numpy()]            # (n_aud,)

    # normalize for display
    a = aud_sal - aud_sal.min()
    a = a / (a.max() + 1e-9)

    # ── map audio tokens to time, overlay on waveform ─────────────────────────
    y, sr = _load_waveform(args.video)
    dur = len(y) / sr
    tok_t = (np.arange(n_aud) + 0.5) / n_aud * dur      # token -> time (s)
    wav_t = np.arange(len(y)) / sr

    # top-k attended segments
    k = min(args.top_k, n_aud)
    top_idx = np.argsort(a)[-k:][::-1]
    top_times = sorted(float(tok_t[i]) for i in top_idx)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), height_ratios=[2, 1],
                                   sharex=True)
    # waveform + attention curve
    ax1.plot(wav_t, y, color="0.7", lw=0.5, zorder=1)
    axt = ax1.twinx()
    axt.fill_between(tok_t, 0, a, color="#e6231e", alpha=0.30, zorder=2)
    axt.plot(tok_t, a, color="#e6231e", lw=1.6, zorder=3, label="audio attention")
    axt.set_ylim(0, 1.05); axt.set_ylabel("attention (norm)", color="#e6231e")
    seg = dur / n_aud
    for t in top_times:
        ax1.axvspan(t - seg, t + seg, color="#ffd000", alpha=0.35, zorder=0)
        ax1.text(t, ax1.get_ylim()[1]*0.92, f"{t:.1f}s", ha="center",
                 fontsize=9, weight="bold", color="#a06a00")
    ax1.set_ylabel("waveform")
    ax1.set_title(f"{stem} — which parts of the audio get the most attention",
                  fontsize=13, weight="bold")

    # attention-over-time strip
    ax2.imshow(a[np.newaxis, :], aspect="auto", cmap="hot",
               extent=[0, dur, 0, 1])
    ax2.set_yticks([]); ax2.set_xlabel("time (s)")
    ax2.set_title("audio attention over time", fontsize=10)

    fig.tight_layout()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}_audio_attention.png", dpi=150)
    plt.close(fig)

    (out_dir / f"{stem}_audio_attention.json").write_text(json.dumps({
        "video": stem, "n_audio": n_aud, "duration_s": dur,
        "top_attended_times_s": top_times,
        "attention_per_token": aud_sal.tolist(),
    }, indent=2))
    print(f"  top attended audio times (s): "
          f"{', '.join(f'{t:.1f}' for t in top_times)}", flush=True)
    print(f"Saved {out_dir / (stem + '_audio_attention.png')}", flush=True)


if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
