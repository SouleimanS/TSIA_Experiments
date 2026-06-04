"""Training driver for AVModelV6 (sink-aware C-MIB) on MUSIC-AVQA.

Usage:
    python -m av_ib.train.train_v6 \
        --ann-path /path/to/avqa-train.json \
        --video-root /path/to/videos/all \
        --num-steps 100 \
        --beta-v 0 --beta-a 0 --beta-j 0 \
        --log-path runs/sanity/log.jsonl \
        --ckpt-path runs/sanity/final.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class MusicAVQAPathDataset(Dataset):
    """Returns file paths + strings — matches QwenOmniWrapper.forward_train signature.

    NOT the same as the old MusicAVQADataset, which returned preprocessed tensors.
    The Qwen wrapper does its own loading via process_mm_info, so we just point at files.
    """

    def __init__(self, ann_path, video_root, skip_deleted=True):
        with open(ann_path) as f:
            records = json.load(f)
        if skip_deleted:
            records = [r for r in records if r.get("question_deleted", 0) == 0]
        # Pre-filter to records whose video files exist (skip broken refs)
        self.video_root = Path(video_root)
        self.records = [r for r in records
                        if (self.video_root / f"{r['video_id']}.mp4").exists()]
        dropped = len(records) - len(self.records)
        if dropped:
            print(f"  Dropped {dropped} records with missing videos.")
        print(f"  Dataset size: {len(self.records)} records.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from av_ib.data.musicavqa import render_question
        rec = self.records[idx]
        video_path = str(self.video_root / f"{rec['video_id']}.mp4")
        prompt = render_question(rec["question_content"], rec["templ_values"])
        answer = rec["anser"]
        return {
            "video": video_path,
            "audio": video_path,   # audio extracted from video by Qwen processor
            "prompt": prompt,
            "answer": answer,
        }


def collate(batch):
    """B=1 collate that just unstacks the dict fields into parallel lists."""
    return {
        "videos": [b["video"] for b in batch],
        "audios": [b["audio"] for b in batch],
        "prompts": [b["prompt"] for b in batch],
        "answers": [b["answer"] for b in batch],
    }


def main(args):
    print("=" * 60)
    print(f"v5 training: {args.num_steps} steps")
    print("=" * 60)

    print("\n[1/3] Constructing AVModelV6...")
    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(use_lora=(not args.no_lora), variant=args.variant, video_vib=args.video_vib, audio_vib=args.audio_vib, fusion_type=args.fusion)

    print("\n[2/3] Building dataset...")
    dataset = MusicAVQAPathDataset(args.ann_path, args.video_root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,   # Qwen processor isn't fork-safe; keep main-process loading
        collate_fn=collate,
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
        model_handles_betas=False,  # v6 returns RAW combined KL; trainer applies beta_v/beta_j as usual
    )

    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta-v", type=float, default=0.0)
    p.add_argument("--beta-a", type=float, default=0.0)
    p.add_argument("--beta-j", type=float, default=0.0)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--log-path", default="train_log.jsonl")
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--no-lora", action="store_true", default=False, help="Disable LoRA on the LLM (full frozen)")
    p.add_argument("--variant", choices=("a", "b", "b_bis", "b_bis_2", "b_bis_3", "b_bis_4", "c"), default="a",
                   help="v6 architecture variant: a=3VIB-sink-everywhere, b=2VIB-no-joint, c=3VIB-sink-on-v-only")
    p.add_argument("--video-vib", choices=("sink", "standard"), default="sink",
                   help="video bottleneck type: sink=SinkAwareVIB, standard=plain VIB (isolation ablation)")
    p.add_argument("--audio-vib", choices=("standard", "norm_topk"), default="standard",
                   help="audio VIB: standard=plain VIB, norm_topk=top-k norm sink protection")
    p.add_argument("--fusion", choices=("mutual", "sink_sym", "none"), default="mutual",
                   help="fusion module: mutual=MutualCrossAttention, sink_sym=sink-mediated symmetric")
    p.add_argument("--save-every", type=int, default=0,
                   help="If >0, save step_N.pt every N steps")
    args = p.parse_args()
    main(args)
