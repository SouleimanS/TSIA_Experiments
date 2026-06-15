"""Training driver for AVModelBalanced."""
from __future__ import annotations

import argparse

import torch

from av_ib.train.loop import run_training


def main(args):
    from av_ib.balanced.model import AVModelBalanced

    print("=" * 60)
    print(f"balanced training | variant={args.variant} | dataset={args.dataset}")
    print(f"  audio_balance={args.audio_balance}  dir_weight={args.dir_weight}")
    print("=" * 60)

    print("\n[1/3] Constructing AVModelBalanced...")
    model = AVModelBalanced(
        use_lora=True,
        variant=args.variant,
        device_map="cuda:0",
        audio_balance=args.audio_balance,
        dir_weight=args.dir_weight,
    ).train()
    model.set_sample_noise(True)

    n_bal = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_bal/1e6:.1f}M")

    print("\n[2/3] Building dataset...")
    if args.dataset == "avqa":
        from av_ib.data.avqa import AVQADataset
        dataset = AVQADataset(args.ann_path, args.video_root)
    elif args.dataset == "avhbench":
        from av_ib.data.avhbench_qa import AVHBenchQADataset
        dataset = AVHBenchQADataset(args.ann_path, args.video_root, split="train")
    else:
        raise ValueError(args.dataset)

    print("\n[3/3] Starting training loop...")
    summary = run_training(
        model=model,
        dataset=dataset,
        num_steps=args.num_steps,
        lr=args.lr,
        beta_v=args.beta_v,
        beta_a=args.beta_a,
        beta_j=0.0,
        aux_weight=args.aux_weight,
        log_path=args.log_path,
        ckpt_path=args.ckpt_path,
        save_every=args.save_every,
        print_every=args.print_every,
    )
    import json
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",     default="avqa", choices=["avqa", "avhbench"])
    ap.add_argument("--ann-path",    required=True)
    ap.add_argument("--video-root",  required=True)
    ap.add_argument("--variant",     default="b_topk_nofusion")
    ap.add_argument("--num-steps",   type=int,   default=2000)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--beta-v",      type=float, default=7e-6)
    ap.add_argument("--beta-a",      type=float, default=7e-6)
    ap.add_argument("--aux-weight",  type=float, default=0.1)
    ap.add_argument("--dir-weight",  type=float, default=0.1,
                    help="Weight for audio direction-preservation loss (0=disable).")
    ap.add_argument("--audio-balance", action="store_true", default=True,
                    help="Scale audio tokens by sqrt(n_v/n_a) (option 1).")
    ap.add_argument("--no-audio-balance", dest="audio_balance", action="store_false")
    ap.add_argument("--log-path",    default="runs/balanced/balanced.jsonl")
    ap.add_argument("--ckpt-path",   default="runs/balanced/balanced_final.pt")
    ap.add_argument("--save-every",  type=int, default=500)
    ap.add_argument("--print-every", type=int, default=50)
    main(ap.parse_args())
