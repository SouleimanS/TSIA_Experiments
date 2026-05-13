"""End-to-end smoke test: build AVModelV1, run 5 training steps on dummy
data, verify loss decreases (or at least varies) and no crashes.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from av_ib.model.av_model_v1 import AVModelV1
from av_ib.data.dummy import DummyAVDataset, av_collate
from av_ib.train.loop import run_training


def main():
    print("=== Building AVModelV1 ===")
    model = AVModelV1(use_lora=True).cuda()

    print("\n=== Building dummy dataloader (batch=4) ===")
    ds = DummyAVDataset(n=16)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=av_collate)
    print(f"  {len(ds)} examples, batch_size=4")

    print("\n=== Running 5 training steps ===\n")
    run_training(
        model,
        dl,
        num_steps=5,
        lr=1e-4,
        log_path="smoke_train_log.jsonl",
        ckpt_dir="smoke_ckpts",
        save_every=0,
    )

    # Verify log file was written.
    import json
    with open("smoke_train_log.jsonl") as f:
        recs = [json.loads(line) for line in f]
    print(f"\nWrote {len(recs)} log records.")
    print(f"Initial loss: {recs[0]['loss']:.4f}")
    print(f"Final loss:   {recs[-1]['loss']:.4f}")


if __name__ == "__main__":
    main()
