"""Validate the architecture can learn a video->answer signal.

Same protocol as test_synthetic_audio.py but with planted signal in
videos instead of audio. Pass criteria identical:
    - Training loss drops below 0.5
    - Held-out accuracy > 70%
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from av_ib.model.av_model_v1 import AVModelV1
from av_ib.data.synthetic_video import SyntheticVideoDataset
from av_ib.data.dummy import av_collate
from av_ib.train.loop import run_training


@torch.no_grad()
def evaluate(model, dataset, batch_size=4):
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=av_collate)
    correct = 0
    total = 0
    for batch in dl:
        videos = batch["videos"].cuda()
        audio_mels = batch["audio_mels"].cuda()
        prompts = batch["prompts"]
        gold = batch["answers"]
        preds = model.forward_generate(videos, audio_mels, prompts, max_new_tokens=5)
        for pred, g in zip(preds, gold):
            pred_yn = "Yes" if "Yes" in pred else ("No" if "No" in pred else "?")
            gold_yn = "Yes" if "Yes" in g else "No"
            correct += (pred_yn == gold_yn)
            total += 1
            print(f"    gold={gold_yn}  pred={pred[:30]!r}  match={pred_yn==gold_yn}")
    model.train()
    return correct / total if total > 0 else 0.0


def main():
    print("=== Building AVModelV1 ===")
    model = AVModelV1(use_lora=True).cuda()

    train_ds = SyntheticVideoDataset(n=64, seed=0, hold_out=False)
    held_ds = SyntheticVideoDataset(n=16, seed=0, hold_out=True)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=av_collate)

    print("\n=== Initial held-out accuracy (untrained) ===")
    acc_before = evaluate(model, held_ds)
    print(f"  accuracy: {acc_before:.2%}\n")

    print("\n=== Training for 100 steps ===\n")
    run_training(
        model,
        train_dl,
        num_steps=100,
        lr=1e-4,
        log_path="synthetic_video_log.jsonl",
        ckpt_dir="synthetic_video_ckpts",
        save_every=0,
        print_every=10,
    )

    print("\n=== Final held-out accuracy ===")
    acc_after = evaluate(model, held_ds)
    print(f"  accuracy: {acc_after:.2%}")

    import json
    with open("synthetic_video_log.jsonl") as f:
        recs = [json.loads(line) for line in f]
    final_loss = recs[-1]["loss"]

    print(f"\nFinal training loss: {final_loss:.4f}")
    print("\n=== Pass/fail ===")
    loss_ok = final_loss < 0.5
    acc_ok = acc_after > 0.70
    print(f"  loss < 0.5:      {loss_ok}  (got {final_loss:.4f})")
    print(f"  held-out > 70%:  {acc_ok}   (got {acc_after:.2%})")
    if loss_ok and acc_ok:
        print("\n  PASS: architecture can learn video->answer signal.")
    elif loss_ok and not acc_ok:
        print("\n  PARTIAL: memorised training set but doesn't generalise.")
    else:
        print("\n  FAIL: model can't fit the training signal.")


if __name__ == "__main__":
    main()
