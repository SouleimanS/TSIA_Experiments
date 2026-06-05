"""Training driver for AVModelV6 on MUSIC-AVQA or AVQA.

Usage:
    python -m av_ib.train.train_v6 \\
        --dataset music_avqa \\
        --ann-path /path/to/avqa-train.json \\
        --video-root /path/to/videos \\
        --variant b_std_fusion \\
        --num-steps 100 \\
        --beta-v 0 --beta-a 0 --beta-j 0 \\
        --log-path runs/sanity/log.jsonl \\
        --ckpt-path runs/sanity/final.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import random

from av_ib.model.av_model_v6 import _ALL_VARIANTS


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def _collate(batch):
    return {
        "videos":  [b["video_path"] for b in batch],
        "audios":  [b["audio_path"] for b in batch],
        "prompts": [b["prompt"]     for b in batch],
        "answers": [b["answer"]     for b in batch],
    }


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_dataset(dataset: str, ann_path: str, video_root: str,
                  max_samples: int = 0, seed: int = 42):
    if dataset == "music_avqa":
        from av_ib.data.musicavqa import MusicAVQADataset
        ds = MusicAVQADataset(ann_path, video_root)
    elif dataset == "avqa":
        from av_ib.data.avqa import AVQADataset
        ds = AVQADataset(ann_path, video_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose music_avqa or avqa.")

    if max_samples > 0 and len(ds) > max_samples:
        from torch.utils.data import Subset
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), max_samples)
        ds = Subset(ds, sorted(indices))
        print(f"  Subset: {max_samples} samples (seed={seed})")

    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("=" * 60)
    print(f"v6 training | variant={args.variant} | dataset={args.dataset} | steps={args.num_steps}")
    print("=" * 60)

    print("\n[1/3] Constructing AVModelV6...")
    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(
        use_lora=(not args.no_lora),
        variant=args.variant,
        adaptive_beta_base=args.adaptive_beta_base,
    )
    if args.no_sample_noise:
        model.set_sample_noise(False)
        print("  reparam noise DISABLED (z = mu) — pilot isolates splice path")

    print("\n[2/3] Building dataset...")
    dataset = build_dataset(args.dataset, args.ann_path, args.video_root,
                            max_samples=args.max_samples, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,   # Qwen processor is not fork-safe
        collate_fn=_collate,
    )

    print("\n[3/3] Starting training loop...")
    from av_ib.train.loop import run_training
    summary = run_training(
        model, loader,
        num_steps=args.num_steps,
        lr=args.lr,
        beta_v=args.beta_v,
        beta_a=args.beta_a,
        beta_j=args.beta_j,
        aux_weight=args.aux_weight,
        log_path=args.log_path,
        ckpt_path=args.ckpt_path,
        print_every=args.print_every,
        save_every=args.save_every,
        model_handles_betas=False,
    )

    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Dataset
    p.add_argument("--dataset", choices=("music_avqa", "avqa"), default="music_avqa",
                   help="Training dataset: music_avqa or avqa (original multi-choice)")
    p.add_argument("--ann-path", required=True,
                   help="Path to annotation JSON (avqa-train.json)")
    p.add_argument("--video-root", required=True,
                   help="Directory containing <video_id>.mp4 files")
    # Model
    p.add_argument("--variant", choices=_ALL_VARIANTS, default="b_std_fusion",
                   help="v6 architecture variant (see av_model_v6 module docstring)")
    p.add_argument("--no-lora", action="store_true", default=False,
                   help="Disable LoRA (full frozen backbone)")
    p.add_argument("--adaptive-beta-base", type=float, default=0.1,
                   help="Base scale for AdaVIB beta (only used by *_adavib variants)")
    # Optimisation
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta-v", type=float, default=0.0)
    p.add_argument("--beta-a", type=float, default=0.0)
    p.add_argument("--beta-j", type=float, default=0.0)
    p.add_argument("--aux-weight", type=float, default=0.1)
    # Logging / checkpoints
    p.add_argument("--log-path", default="train_log.jsonl")
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0,
                   help="Save step_N.pt every N steps (0 = disabled)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap dataset size to N randomly selected samples (0 = use all)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for subset selection")
    p.add_argument("--no-sample-noise", action="store_true", default=False,
                   help="Disable reparam noise (z=mu) to isolate the splice path "
                        "from injected VIB noise in a beta=0 pilot")
    args = p.parse_args()
    main(args)
