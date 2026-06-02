"""Extract sink dimensions (Dsink) from Qwen3-Omni Thinker.

Following Kang et al. 2025 / KAIST ASD paper: massive-activation dimensions
of the BOS-equivalent token are stable across layers and act as sink
characterization dimensions.

Two passes:
  (1) text-only via thinker.model (clean BOS-only activation)
  (2) multimodal via thinker forward with a real MUSIC-AVQA record

Outputs:
  - dsink_text_only.json   : top-K dims per layer (text-only)
  - dsink_multimodal.json  : top-K dims per layer (multimodal context)
  - dsink_analysis.json    : comparison + recommended Dsink

Usage:
  python -m av_ib.eval.extract_dsink \\
      --record-video-path /path/to/some/sample.mp4 \\
      --record-prompt "What is happening?" \\
      --top-k 4 \\
      --out-dir results/dsink/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


DEFAULT_BOS_TOKEN_ID = 151643   # <|endoftext|>
TOP_K = 4


def rmsnorm(x, eps=1e-6):
    """RMSNorm along the last dimension."""
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def find_top_k_dims(hidden_state_at_token, top_k=4):
    """Given (D,) tensor for one token, return top-K indices and magnitude stats."""
    norm = rmsnorm(hidden_state_at_token.float())
    abs_norm = norm.abs()
    top_values, top_indices = torch.topk(abs_norm, k=top_k)
    median_norm = abs_norm.median().item()
    return top_indices.cpu().tolist(), top_values.cpu().tolist(), median_norm


def text_only_forward(model, bos_id):
    """Pass 1: feed only [BOS] through thinker.model directly."""
    print("\n=== PASS 1: text-only thinker.model([BOS]) ===")
    thinker_text = model.thinker.model
    device = next(thinker_text.parameters()).device

    input_ids = torch.tensor([[bos_id]], device=device)
    with torch.no_grad():
        out = thinker_text(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    # out.hidden_states is a tuple of (L+1) tensors, each (1, 1, D)
    # First is the embedding, then one per layer
    layer_dims = []
    for layer_idx, hs in enumerate(out.hidden_states):
        # hs shape: (1, seq=1, D). We want the single token's activation
        token_act = hs[0, 0, :]
        top_idx, top_val, median_norm = find_top_k_dims(token_act, top_k=TOP_K)
        layer_dims.append({
            "layer": layer_idx,
            "top_dims": top_idx,
            "top_values": top_val,
            "max_norm_value": max(top_val),
            "median_norm": median_norm,
            "ratio_max_to_median": max(top_val) / max(median_norm, 1e-9),
        })

    print(f"  Layers analyzed: {len(layer_dims)}")
    print(f"  Top-4 dims at layer 0: {layer_dims[0]['top_dims']}")
    print(f"  Top-4 dims at last layer: {layer_dims[-1]['top_dims']}")
    return layer_dims


def multimodal_forward(model, processor, video_path, prompt, bos_id):
    """Pass 2: full Thinker forward on a real record. Find BOS token, extract."""
    print("\n=== PASS 2: multimodal thinker(...) end-to-end ===")
    from qwen_omni_utils import process_mm_info
    from av_ib.backbone.qwen_omni import _build_conversation

    conv = _build_conversation(video_path, prompt, answer=None)
    text = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False,
    )
    audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    ).to(model.device).to(model.dtype)

    # Find positions of BOS in input_ids
    input_ids = inputs["input_ids"][0]
    bos_positions = (input_ids == bos_id).nonzero(as_tuple=True)[0]
    print(f"  Input length: {input_ids.shape[0]}, BOS positions: {bos_positions.tolist()}")
    if len(bos_positions) == 0:
        print(f"  WARNING: token id {bos_id} not found in input. Skipping multimodal pass.")
        return None

    bos_pos = bos_positions[0].item()   # Use first occurrence

    with torch.no_grad():
        out = model.thinker(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    layer_dims = []
    for layer_idx, hs in enumerate(out.hidden_states):
        # hs shape: (1, seq_len, D). Get BOS position activation.
        token_act = hs[0, bos_pos, :]
        top_idx, top_val, median_norm = find_top_k_dims(token_act, top_k=TOP_K)
        layer_dims.append({
            "layer": layer_idx,
            "top_dims": top_idx,
            "top_values": top_val,
            "max_norm_value": max(top_val),
            "median_norm": median_norm,
            "ratio_max_to_median": max(top_val) / max(median_norm, 1e-9),
        })

    print(f"  Layers analyzed: {len(layer_dims)}")
    print(f"  Top-4 dims at layer 0: {layer_dims[0]['top_dims']}")
    print(f"  Top-4 dims at last layer: {layer_dims[-1]['top_dims']}")
    return layer_dims


def consensus_dsink(text_layers, mm_layers, top_k=4):
    """Compute consensus Dsink across all layers (and both passes)."""
    from collections import Counter
    counter = Counter()
    for entry in text_layers:
        counter.update(entry["top_dims"])
    if mm_layers is not None:
        for entry in mm_layers:
            counter.update(entry["top_dims"])

    # Sort by frequency descending
    most_common = counter.most_common()
    print(f"\n=== Consensus Dsink (sorted by cross-layer frequency) ===")
    for dim, count in most_common[:20]:
        print(f"  dim {dim:5d}: appears in {count} (layer, pass) combos")

    # Take top-K consistently
    consensus = [dim for dim, _ in most_common[:top_k]]
    print(f"\n  Recommended Dsink (top-{top_k}): {consensus}")
    return consensus, most_common


def main(args):
    bos_id = args.bos_id
    print("=" * 60)
    print("Dsink extraction for Qwen3-Omni Thinker")
    print(f"  BOS token id: {bos_id}")
    print(f"  Top-K dims:   {args.top_k}")
    print("=" * 60)

    print("\n[1/3] Loading model...")
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
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\n[2/3] Running text-only forward...")
    text_layers = text_only_forward(model, bos_id)

    print("\n[3/3] Running multimodal forward...")
    mm_layers = None
    if args.record_video_path:
        try:
            mm_layers = multimodal_forward(
                model, processor, args.record_video_path, args.record_prompt,
                bos_id,
            )
        except Exception as e:
            print(f"  Multimodal forward failed: {type(e).__name__}: {e}")
            print(f"  Continuing with text-only Dsink.")
    else:
        print(f"  Skipping (no --record-video-path)")

    # Magnitude summary
    print(f"\n=== Magnitude check: max/median ratio per layer (text-only) ===")
    for entry in text_layers:
        print(f"  layer {entry['layer']:2d}: max={entry['max_norm_value']:.3f}  "
              f"median={entry['median_norm']:.4f}  "
              f"ratio={entry['ratio_max_to_median']:.1f}x")

    consensus, freq_table = consensus_dsink(text_layers, mm_layers, top_k=args.top_k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "dsink_text_only.json", "w") as f:
        json.dump({"layers": text_layers, "bos_id": bos_id}, f, indent=2)

    if mm_layers is not None:
        with open(out_dir / "dsink_multimodal.json", "w") as f:
            json.dump({"layers": mm_layers, "bos_id": bos_id}, f, indent=2)

    with open(out_dir / "dsink_analysis.json", "w") as f:
        json.dump({
            "recommended_dsink": consensus,
            "frequency_table": [
                {"dim": d, "count": c} for d, c in freq_table[:50]
            ],
            "top_k": args.top_k,
            "bos_id": bos_id,
        }, f, indent=2)

    # Save Dsink as a tensor too for downstream code
    torch.save(torch.tensor(consensus, dtype=torch.long), out_dir / "dsink.pt")

    print(f"\n=== Saved to {out_dir} ===")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record-video-path", default=None,
                   help="Path to a sample video for multimodal forward (optional)")
    p.add_argument("--record-prompt", default="What is happening in this video?",
                   help="Prompt for the multimodal forward")
    p.add_argument("--bos-id", type=int, default=DEFAULT_BOS_TOKEN_ID,
                   help="Token ID to extract Dsink from (151643=<|endoftext|>, 151644=<|im_start|>)")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--out-dir", default="results/dsink")
    args = p.parse_args()
    sys.exit(main(args))
