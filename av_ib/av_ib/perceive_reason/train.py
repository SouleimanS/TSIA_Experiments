"""Training driver for AVModelPerceive (perceive-then-reason C-MIB).

Mirrors av_ib.modality_gate.train but builds AVModelPerceive, which adds a
frozen perception pass (video description via Qwen) and a CrossModalAttention
module that lets the VIB-processed AV tokens attend to the description
embeddings. Only trained: cross_attn, LoRA weights, VIB weights.

The existing run_training loop is reused unchanged since forward_train returns
the same 6-tuple as AVModelV6.

Usage:
    python -m av_ib.perceive_reason.train \\
        --dataset avqa \\
        --ann-path /path/to/avqa-train.json \\
        --video-root /path/to/videos \\
        --variant b_topk_nofusion \\
        --num-steps 2000 \\
        --beta-v 7e-6 --beta-a 7e-6 \\
        --n-perceive-frames 4 \\
        --cross-attn-heads 8 \\
        --cross-attn-dropout 0.1 \\
        --log-path runs/perceive/log.jsonl \\
        --ckpt-path runs/perceive/final.pt
"""
from __future__ import annotations

import argparse
import json

from torch.utils.data import DataLoader

from av_ib.model.av_model_v6 import _ALL_VARIANTS
from av_ib.train.train_v6 import _collate, build_dataset
from av_ib.train.loop import run_training


def main(args):
    print("=" * 60)
    print(f"perceive-reason training | variant={args.variant} | "
          f"dataset={args.dataset} | steps={args.num_steps}")
    print("=" * 60)

    print("\n[1/3] Constructing AVModelPerceive...")
    from av_ib.perceive_reason.model import AVModelPerceive
    model = AVModelPerceive(
        use_lora=(not args.no_lora),
        variant=args.variant,
        adaptive_beta_base=args.adaptive_beta_base,
        n_perceive_frames=args.n_perceive_frames,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_dropout=args.cross_attn_dropout,
    )
    if args.no_sample_noise:
        model.set_sample_noise(False)
        print("  reparam noise DISABLED (z = mu)")

    n_cross = sum(p.numel() for p in model.cross_attn.parameters())
    print(f"  cross_attn params: {n_cross / 1e3:.1f}K  "
          f"(heads={args.cross_attn_heads}, dropout={args.cross_attn_dropout})")

    print("\n[2/3] Building dataset...")
    dataset = build_dataset(
        args.dataset, args.ann_path, args.video_root,
        max_samples=args.max_samples, seed=args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=_collate,
    )

    print("\n[3/3] Starting training loop...")
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
    p.add_argument("--dataset", choices=("music_avqa", "avqa"), default="avqa")
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    # Model
    p.add_argument("--variant", choices=_ALL_VARIANTS, default="b_topk_nofusion")
    p.add_argument("--no-lora", action="store_true", default=False)
    p.add_argument("--adaptive-beta-base", type=float, default=0.1)
    # Perception
    p.add_argument("--n-perceive-frames", type=int, default=4,
                   help="Number of frames sampled for the perception pass.")
    # Cross-modal attention
    p.add_argument("--cross-attn-heads", type=int, default=8,
                   help="Number of heads in CrossModalAttention.")
    p.add_argument("--cross-attn-dropout", type=float, default=0.1,
                   help="Dropout in CrossModalAttention (train only).")
    # Optimisation
    p.add_argument("--num-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta-v", type=float, default=7e-6)
    p.add_argument("--beta-a", type=float, default=7e-6)
    p.add_argument("--beta-j", type=float, default=0.0)
    p.add_argument("--aux-weight", type=float, default=0.1)
    # Logging / checkpoints
    p.add_argument("--log-path", default="runs/perceive/train_log.jsonl")
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-sample-noise", action="store_true", default=False)
    args = p.parse_args()
    main(args)
