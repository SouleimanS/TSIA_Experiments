"""Training driver for AVModelGated (modality-gated C-MIB).

Mirrors av_ib.train.train_v6 but builds AVModelGated, which adds the
prompt-conditioned modality gate. The gate trains end-to-end through the
answer NLL (no extra loss term), so the existing run_training loop is reused
unchanged. The gate distribution (p_video / p_audio / p_text) is logged per
step via the model's diagnostics.

Usage:
    python -m av_ib.modality_gate.train \\
        --dataset avqa \\
        --ann-path /path/to/avqa-train.json \\
        --video-root /path/to/videos \\
        --variant b_topk_nofusion \\
        --num-steps 500 \\
        --beta-v 0 --beta-a 0 --beta-j 0 \\
        --log-path runs/gate/log.jsonl \\
        --ckpt-path runs/gate/final.pt
"""
from __future__ import annotations

import argparse
import json
import random

from torch.utils.data import DataLoader

from av_ib.model.av_model_v6 import _ALL_VARIANTS
from av_ib.train.train_v6 import _collate, build_dataset
from av_ib.train.loop import run_training


def main(args):
    print("=" * 60)
    print(f"gated training | variant={args.variant} | dataset={args.dataset} "
          f"| steps={args.num_steps}")
    print("=" * 60)

    print("\n[1/3] Constructing AVModelGated...")
    from av_ib.modality_gate.model import AVModelGated
    model = AVModelGated(
        use_lora=(not args.no_lora),
        variant=args.variant,
        adaptive_beta_base=args.adaptive_beta_base,
        gate_hidden=args.gate_hidden,
        gate_centered=(not args.gate_raw),
    )
    if args.no_sample_noise:
        model.set_sample_noise(False)
        print("  reparam noise DISABLED (z = mu)")

    n_gate = sum(p.numel() for p in model.gate.parameters())
    print(f"  gate params: {n_gate/1e3:.1f}K  "
          f"(centered={not args.gate_raw}, hidden={args.gate_hidden})")

    print("\n[2/3] Building dataset...")
    dataset = build_dataset(args.dataset, args.ann_path, args.video_root,
                            max_samples=args.max_samples, seed=args.seed)
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
    # Gate
    p.add_argument("--gate-hidden", type=int, default=512,
                   help="Hidden width of the modality-gate MLP")
    p.add_argument("--gate-raw", action="store_true", default=False,
                   help="Scale by raw p instead of centred (M*p). Ablation only.")
    # Optimisation
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta-v", type=float, default=0.0)
    p.add_argument("--beta-a", type=float, default=0.0)
    p.add_argument("--beta-j", type=float, default=0.0)
    p.add_argument("--aux-weight", type=float, default=0.1)
    # Logging / checkpoints
    p.add_argument("--log-path", default="runs/gate/train_log.jsonl")
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-sample-noise", action="store_true", default=False)
    args = p.parse_args()
    main(args)
