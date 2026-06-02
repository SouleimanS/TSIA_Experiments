"""Evaluate a trained AVModelV5 checkpoint on MUSIC-AVQA val.

Loads the trainable_state from a checkpoint (~710M params: LoRA + VIBs +
fusion + aux heads) on top of the pretrained Qwen3-Omni base (30B frozen).
Runs generation on N val records, parses, scores, writes CSV.

Usage:
    python -m av_ib.eval.v5_eval \
        --ckpt-path runs/sanity_b0/final.pt \
        --ann-path /path/to/avqa-test.json \
        --video-root /path/to/videos/all \
        --num-records 50 \
        --seed 42 \
        --out-csv runs/sanity_b0/eval_50.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch


from av_ib.eval.answer_parser import parse_answer


def main(args):
    print("=" * 60)
    print(f"v5 eval: {args.num_records} records from {args.ann_path}")
    print(f"  ckpt: {args.ckpt_path}")
    print("=" * 60)

    print("\n[1/4] Constructing AVModelV5...")
    from av_ib.model.av_model_v5 import AVModelV5
    model = AVModelV5(use_lora=True)
    model.eval()

    print("\n[2/4] Loading trained checkpoint...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["trainable_state"]
    print(f"  Checkpoint has {len(state)} trainable tensors")
    # strict=False: pretrained 30B stays as-is, only matching keys overwritten
    missing, unexpected = model.load_state_dict(state, strict=False)
    matched = len(state) - len(unexpected)
    print(f"  Loaded:     {matched} tensors")
    print(f"  Unexpected: {len(unexpected)} (problem if > 0)")
    print(f"  Missing:    {len(missing)} (expected — frozen pretrained params)")
    if unexpected:
        print(f"  First unexpected keys: {unexpected[:3]}")

    print("\n[3/4] Sampling val records...")
    with open(args.ann_path) as f:
        records = json.load(f)
    records = [r for r in records if r.get("question_deleted", 0) == 0]
    video_root = Path(args.video_root)
    records = [r for r in records
               if (video_root / f"{r['video_id']}.mp4").exists()]
    print(f"  Pool: {len(records)} records with valid videos")

    rng = random.Random(args.seed)
    picked = rng.sample(records, args.num_records)
    print(f"  Sampled: {len(picked)} records (seed={args.seed})")

    print("\n[4/4] Running generation + scoring...")
    from av_ib.data.musicavqa import render_question
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    n_unparseable = 0
    n_errors = 0
    t0 = time.time()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "video_id", "question", "gold", "raw_pred",
                         "parsed_pred", "correct"])

        for idx, rec in enumerate(picked):
            video_path = str(video_root / f"{rec['video_id']}.mp4")
            prompt = render_question(rec["question_content"], rec["templ_values"])
            gold = rec["anser"].strip().lower()

            try:
                with torch.no_grad():
                    raws = model.forward_generate([video_path], [video_path], [prompt])
                    raw = raws[0]
            except Exception as e:
                import traceback
                if idx == 0:
                    print("\n" + "="*60, flush=True)
                    print(f"FULL TRACEBACK for record 0:", flush=True)
                    traceback.print_exc()
                    print("="*60 + "\n", flush=True)
                raw = f"<ERROR: {type(e).__name__}: {str(e)[:80]}>"
                n_errors += 1

            parsed = parse_answer(raw)
            is_correct = int(parsed == gold)
            correct += is_correct
            if parsed == "??":
                n_unparseable += 1

            writer.writerow([idx, rec["video_id"], rec["question_content"],
                             gold, raw, parsed, is_correct])
            f.flush()

            if (idx + 1) % 5 == 0 or idx == 0:
                acc = 100 * correct / (idx + 1)
                rate = (idx + 1) / (time.time() - t0)
                print(f"  [{idx+1:3d}/{args.num_records}]  "
                      f"acc={acc:.1f}%  rate={rate:.2f}/s  "
                      f"err={n_errors}  unp={n_unparseable}",
                      flush=True)

    elapsed = time.time() - t0
    acc = 100 * correct / args.num_records
    print(f"\n{'='*60}")
    print(f"v5 eval done: {correct}/{args.num_records} = {acc:.2f}%")
    print(f"  Errors:      {n_errors}")
    print(f"  Unparseable: {n_unparseable}")
    print(f"  Elapsed:     {elapsed:.1f}s ({args.num_records/elapsed:.2f}/s)")
    print(f"  Output:      {out_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--num-records", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", default="v5_eval.csv")
    args = p.parse_args()
    sys.exit(main(args))
