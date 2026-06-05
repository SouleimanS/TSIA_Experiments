"""Training driver for QwenLoRABaseline on MUSIC-AVQA or AVQA.

Trains Qwen3-Omni with LoRA only — no VIB, no fusion, no C-MIB components.
Use this to produce the "Qwen3-Omni trained" comparison checkpoint.

Usage:
    python -m av_ib.train.train_baseline \\
        --dataset music_avqa \\
        --ann-path /path/to/avqa-train.json \\
        --video-root /path/to/videos \\
        --num-steps 63854 \\
        --log-path runs/baseline_musicavqa/log.jsonl \\
        --ckpt-path runs/baseline_musicavqa/final.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from av_ib.train.train_v6 import build_dataset, _collate


def main(args):
    print("=" * 60)
    print(f"Qwen3-Omni LoRA baseline | dataset={args.dataset} | steps={args.num_steps}")
    print("=" * 60)

    print("\n[1/3] Constructing QwenLoRABaseline...")
    from av_ib.model.qwen_lora_baseline import QwenLoRABaseline
    model = QwenLoRABaseline(use_lora=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    print("\n[2/3] Building dataset...")
    dataset = build_dataset(args.dataset, args.ann_path, args.video_root,
                            max_samples=args.max_samples, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )

    print("\n[3/3] Starting training loop...")
    from av_ib.train.loop import run_training
    summary = run_training(
        model, loader,
        num_steps=args.num_steps,
        lr=args.lr,
        beta_v=0.0,
        beta_a=0.0,
        beta_j=0.0,
        aux_weight=0.0,   # no aux heads on baseline
        log_path=args.log_path,
        ckpt_path=args.ckpt_path,
        print_every=args.print_every,
        save_every=args.save_every,
    )

    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("music_avqa", "avqa"), default="music_avqa")
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--num-steps", type=int, default=63854)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--log-path", default="train_log.jsonl")
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
