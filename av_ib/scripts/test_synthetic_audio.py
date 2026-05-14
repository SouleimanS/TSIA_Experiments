"""Validate the architecture can learn an audio->answer signal.

Train v1 for N steps on SyntheticAudioDataset, then evaluate on a
held-out set of the same construction. Pass criteria:
    - Training loss drops below 0.5 by the end
    - Held-out accuracy > 70%

If both pass, the architecture routes audio information through to the
answer correctly. If loss drops but accuracy stays at 50%, the model is
memorising example IDs through some shortcut (a bug).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from av_ib.model.av_model_v1 import AVModelV1
from av_ib.data.synthetic_audio import SyntheticAudioDataset
from av_ib.data.dummy import av_collate
from av_ib.train.loop import run_training


@torch.no_grad()
def evaluate(model, dataset, batch_size=4):
    """Generate answers for each example, check exact-match against ground truth."""
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
            # Look for the gold answer word ('Yes' or 'No') in the generation
            pred_yn = "Yes" if "Yes" in pred else ("No" if "No" in pred else "?")
            gold_yn = "Yes" if "Yes" in g else "No"
            correct += (pred_yn == gold_yn)
            total += 1
            print(f"    gold={gold_yn}  pred='{pred[:30]}...'  match={pred_yn==gold_yn}")
    model.train()
    return correct / total if total > 0 else 0.0


def main():
    print("=== Building AVModelV1 ===")
    model = AVModelV1(use_lora=True).cuda()

    print("\n=== Building training set (64 examples) and held-out set (16) ===")
    train_ds = SyntheticAudioDataset(n=64, seed=0, hold_out=False)
    held_ds = SyntheticAudioDataset(n=16, seed=0, hold_out=True)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=av_collate)

    print("\n=== Initial held-out accuracy (untrained model) ===")
    acc_before = evaluate(model, held_ds)
    print(f"  accuracy: {acc_before:.2%}\n")

    print("\n=== Training for 100 steps (lr=1e-4, batch=4) ===\n")
    run_training(
        model,
        train_dl,
        num_steps=100,
        lr=1e-4,
        log_path="synthetic_audio_log.jsonl",
        ckpt_dir="synthetic_audio_ckpts",
        save_every=0,
        print_every=10,
    )

    print("\n=== Final held-out accuracy ===")
    acc_after = evaluate(model, held_ds)
    print(f"  accuracy: {acc_after:.2%}")

    # Read back log for final loss
    import json
    with open("synthetic_audio_log.jsonl") as f:
        recs = [json.loads(line) for line in f]
    final_loss = recs[-1]["loss"]
    print(f"\nFinal training loss: {final_loss:.4f}")

    print("\n=== Pass/fail ===")
    loss_ok = final_loss < 0.5
    acc_ok = acc_after > 0.70
    print(f"  loss < 0.5:      {loss_ok}  (got {final_loss:.4f})")
    print(f"  held-out > 70%:  {acc_ok}   (got {acc_after:.2%})")
    if loss_ok and acc_ok:
        print("\n  PASS: architecture can learn audio->answer signal.")
    elif loss_ok and not acc_ok:
        print("\n  PARTIAL: model memorised training set but doesn't generalise.")
        print("  This suggests a bug: the model is learning a shortcut, not")
        print("  the planted signal. Worth investigating before real training.")
    else:
        print("\n  FAIL: model can't even fit the training signal.")
        print("  Architectural bug likely.")


if __name__ == "__main__":
    main()
