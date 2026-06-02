"""Phase A Part 2: Validate Dsink as encoder-output sink predictor.

Hypothesis: tokens with high φ(x) = max_{d ∈ Dsink} |RMSNorm(x)[d]| at the
LLM input (after embedding, before any transformer layer) will become sinks
in deeper LLM layers (high sink frequency across layers).

If the correlation is strong, encoder-output Dsink projection is a viable
proxy for LLM-internal sink-ness → single-pass sink-aware VIB is possible.
If correlation is weak, sink-ness emerges from transformer processing and
we need the two-pass approach.

Method:
  For N records:
    1. Forward pass through thinker, output_hidden_states=True, output_attentions=False
    2. φ_encoder[token] = max_{d ∈ Dsink} |RMSNorm(input_embeds[token])[d]|
    3. For each layer l: sink_l[token] = 1 if φ(hidden_l[token]) ≥ τ
    4. global_sink_freq[token] = Σ_l sink_l[token]
  Compute:
    - Spearman ρ(φ_encoder, global_sink_freq)
    - Recall@K: of top-K by global_sink_freq, what fraction are in top-K by φ_encoder
    - Same broken down by modality (text/audio/video)

Usage:
  python -m av_ib.eval.validate_dsink_proxy \\
      --ann-path /path/to/avqa-test.json \\
      --video-root /path/to/videos \\
      --dsink-pt results/dsink_imstart/dsink.pt \\
      --dsink-dims 985 1992 \\
      --num-records 100 \\
      --tau 5.0 \\
      --top-k-frac 0.05 \\
      --out-dir results/validate_dsink/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def rmsnorm(x, eps=1e-6):
    """RMSNorm along the last dimension."""
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def compute_phi(hidden_state, dsink_dims):
    """φ(x) = max_{d ∈ Dsink} |RMSNorm(x)[d]| per token.

    hidden_state: (seq_len, D) tensor
    dsink_dims: list of int dim indices
    Returns: (seq_len,) tensor of float phi values
    """
    normed = rmsnorm(hidden_state.float())   # (seq_len, D)
    dsink_proj = normed[:, dsink_dims].abs()  # (seq_len, |Dsink|)
    phi = dsink_proj.max(dim=-1).values       # (seq_len,)
    return phi


def get_modality_mask(input_ids, tokenizer, processor):
    """Return per-token modality label.

    Heuristic for Qwen3-Omni: distinguish text vs media placeholders by
    checking against known placeholder token ids.
    Returns: list of str ['text', 'audio', 'video'] per token position.
    """
    # Best-effort: query the tokenizer for placeholder ids
    placeholder_to_modality = {}
    for special in tokenizer.special_tokens_map.values():
        if isinstance(special, str):
            # Some Qwen variants name them with 'audio' / 'video'
            sp_lower = special.lower()
            if 'audio' in sp_lower:
                ids = tokenizer.encode(special, add_special_tokens=False)
                for i in ids:
                    placeholder_to_modality[i] = 'audio'
            elif 'video' in sp_lower or 'vision' in sp_lower or 'image' in sp_lower:
                ids = tokenizer.encode(special, add_special_tokens=False)
                for i in ids:
                    placeholder_to_modality[i] = 'video'

    # Also check additional_special_tokens
    if hasattr(tokenizer, 'additional_special_tokens'):
        for special in tokenizer.additional_special_tokens or []:
            sp_lower = special.lower()
            if 'audio' in sp_lower:
                ids = tokenizer.encode(special, add_special_tokens=False)
                for i in ids:
                    placeholder_to_modality[i] = 'audio'
            elif 'video' in sp_lower or 'vision' in sp_lower or 'image' in sp_lower:
                ids = tokenizer.encode(special, add_special_tokens=False)
                for i in ids:
                    placeholder_to_modality[i] = 'video'

    ids_list = input_ids.tolist()
    return [placeholder_to_modality.get(i, 'text') for i in ids_list]


def main(args):
    print("=" * 60)
    print("Dsink proxy validation")
    print(f"  Dsink dims: {args.dsink_dims}")
    print(f"  Records:    {args.num_records}")
    print(f"  Tau:        {args.tau}")
    print(f"  Top-K frac: {args.top_k_frac}")
    print("=" * 60)

    print("\n[1/4] Loading model...")
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\n[2/4] Sampling records...")
    with open(args.ann_path) as f:
        records = json.load(f)
    video_root = Path(args.video_root)
    records = [r for r in records
               if r.get("question_deleted", 0) == 0
               and (video_root / f"{r['video_id']}.mp4").exists()]
    rng = random.Random(args.seed)
    picked = rng.sample(records, args.num_records)
    print(f"  Sampled {len(picked)} records (seed={args.seed})")

    print("\n[3/4] Running forwards and collecting φ values...")
    from qwen_omni_utils import process_mm_info
    from av_ib.backbone.qwen_omni import _build_conversation

    dsink_dims = torch.tensor(args.dsink_dims, dtype=torch.long)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-record aggregated stats (we don't store full per-token data — too big)
    # Instead, compute per-record statistics on the fly.
    per_record_stats = []

    for idx, rec in enumerate(picked):
        video_path = str(video_root / f"{rec['video_id']}.mp4")
        prompt = rec["question_content"]

        try:
            t1 = time.time()
            conv = _build_conversation(video_path, prompt, answer=None)
            text = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False,
            )
            audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
            inputs = processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=True,
            ).to(model.device).to(model.dtype)

            with torch.no_grad():
                out = model.thinker(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )

            # hidden_states: tuple of (L+1) tensors, each (B=1, seq_len, D)
            # [0] = embedding output (LLM input), [1..L] = after each layer
            embed = out.hidden_states[0][0]   # (seq_len, D)
            seq_len = embed.shape[0]

            # φ at encoder output (LLM input)
            phi_encoder = compute_phi(embed, dsink_dims.to(embed.device))  # (seq_len,)

            # Per-layer sink count per token. Hidden states may live on
            # different GPUs (device_map="auto" shards layers); move to the
            # accumulator's device.
            sink_device = embed.device
            sink_freq = torch.zeros(seq_len, dtype=torch.int32, device=sink_device)
            for l_idx in range(1, len(out.hidden_states)):
                hs = out.hidden_states[l_idx][0]  # (seq_len, D), arbitrary device
                phi_l = compute_phi(hs, dsink_dims.to(hs.device))
                sink_freq += (phi_l >= args.tau).int().to(sink_device)

            phi_encoder_cpu = phi_encoder.cpu().numpy()
            sink_freq_cpu = sink_freq.cpu().numpy()
            modalities = get_modality_mask(inputs["input_ids"][0], tokenizer, processor)

            # Per-record stats: spearman correlation, recall@K
            from scipy.stats import spearmanr
            rho, p = spearmanr(phi_encoder_cpu, sink_freq_cpu)
            top_k = max(1, int(seq_len * args.top_k_frac))
            top_phi_idx = set(np.argsort(-phi_encoder_cpu)[:top_k])
            top_sink_idx = set(np.argsort(-sink_freq_cpu)[:top_k])
            recall_at_k = len(top_phi_idx & top_sink_idx) / top_k

            # Per-modality breakdown
            mod_stats = {}
            for mod_name in ('text', 'audio', 'video'):
                mask = [i for i, m in enumerate(modalities) if m == mod_name]
                if len(mask) < 2:
                    continue
                phi_m = phi_encoder_cpu[mask]
                sf_m = sink_freq_cpu[mask]
                if len(set(sf_m)) > 1 and len(set(phi_m)) > 1:
                    rho_m, _ = spearmanr(phi_m, sf_m)
                else:
                    rho_m = float('nan')
                mod_stats[mod_name] = {
                    "n_tokens": len(mask),
                    "rho": float(rho_m),
                    "mean_phi": float(phi_m.mean()),
                    "mean_sink_freq": float(sf_m.mean()),
                }

            stats = {
                "idx": idx,
                "video_id": rec["video_id"],
                "seq_len": seq_len,
                "n_layers_after_embed": len(out.hidden_states) - 1,
                "rho_all_tokens": float(rho),
                "rho_p_value": float(p),
                "recall_at_top_k_frac": recall_at_k,
                "top_k": top_k,
                "max_sink_freq": int(sink_freq_cpu.max()),
                "mean_sink_freq": float(sink_freq_cpu.mean()),
                "max_phi_encoder": float(phi_encoder_cpu.max()),
                "mean_phi_encoder": float(phi_encoder_cpu.mean()),
                "modality_breakdown": mod_stats,
                "elapsed_s": time.time() - t1,
            }
            per_record_stats.append(stats)

            if (idx + 1) % 5 == 0 or idx == 0:
                avg_rho = np.mean([s["rho_all_tokens"] for s in per_record_stats])
                avg_recall = np.mean([s["recall_at_top_k_frac"] for s in per_record_stats])
                print(f"  [{idx+1:3d}/{args.num_records}]  "
                      f"avg_ρ={avg_rho:.3f}  avg_recall@K={avg_recall:.3f}  "
                      f"({stats['elapsed_s']:.1f}s/rec)",
                      flush=True)

            del out, embed, sink_freq
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"  Record {idx}: ERROR {type(e).__name__}: {e}")
            if idx < 3:
                traceback.print_exc()

    print("\n[4/4] Final summary")
    rho_all = [s["rho_all_tokens"] for s in per_record_stats]
    recall_all = [s["recall_at_top_k_frac"] for s in per_record_stats]
    print(f"  Records processed:    {len(per_record_stats)}")
    print(f"  Spearman ρ:")
    print(f"    mean ± std:         {np.mean(rho_all):.3f} ± {np.std(rho_all):.3f}")
    print(f"    median:             {np.median(rho_all):.3f}")
    print(f"    range:              [{np.min(rho_all):.3f}, {np.max(rho_all):.3f}]")
    print(f"  Recall@top-{int(args.top_k_frac*100)}%:")
    print(f"    mean ± std:         {np.mean(recall_all):.3f} ± {np.std(recall_all):.3f}")
    print(f"    median:             {np.median(recall_all):.3f}")

    # Per-modality aggregate
    for mod in ('text', 'audio', 'video'):
        mod_rhos = [s["modality_breakdown"].get(mod, {}).get("rho")
                    for s in per_record_stats]
        mod_rhos = [r for r in mod_rhos if r is not None and not np.isnan(r)]
        if mod_rhos:
            print(f"  {mod} tokens ρ:        mean={np.mean(mod_rhos):.3f}  n={len(mod_rhos)}")

    with open(out_dir / "per_record_stats.json", "w") as f:
        json.dump(per_record_stats, f, indent=2)
    summary = {
        "dsink_dims": args.dsink_dims,
        "tau": args.tau,
        "top_k_frac": args.top_k_frac,
        "n_records": len(per_record_stats),
        "rho_mean": float(np.mean(rho_all)),
        "rho_std": float(np.std(rho_all)),
        "rho_median": float(np.median(rho_all)),
        "recall_mean": float(np.mean(recall_all)),
        "recall_std": float(np.std(recall_all)),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Saved to {out_dir} ===")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--dsink-dims", type=int, nargs="+", default=[985, 1992])
    p.add_argument("--num-records", type=int, default=5)
    p.add_argument("--tau", type=float, default=5.0,
                   help="Threshold for sink classification at each layer")
    p.add_argument("--top-k-frac", type=float, default=0.05,
                   help="Fraction of tokens to consider 'top sinks'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/validate_dsink/")
    args = p.parse_args()
    sys.exit(main(args))
