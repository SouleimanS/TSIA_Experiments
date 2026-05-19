"""Smoke test on real AVQA data.

Builds AVModelV1, runs 2 training steps on 4 real AVQA yes/no items.
Verifies loss is finite and shapes flow through the full pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Subset

from av_ib.model.av_model_v1 import AVModelV1
from av_ib.data.avqa import AVQADataset
from av_ib.data.dummy import av_collate
from av_ib.train.loop import run_training


ANN = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_train.json"
VID = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVQA" / "videos" / "Train"


def main():
    print("=== Building AVModelV1 on cuda ===")
    model = AVModelV1(use_lora=True).cuda()
    summary = model.trainable_summary()
    print(f"  Trainable params: {summary['__total_trainable__']:,}")
    print(f"  Total params:     {summary['__total_params__']:,}")
    for k, v in summary.items():
        if k.startswith("__"): continue
        print(f"    {k}: {v:,}")

    print("\n=== Building AVQA dataset (first 4 items) ===")
    full = AVQADataset(ann_path=ANN, video_root=VID)
    ds = Subset(full, list(range(4)))
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=av_collate, num_workers=0)
    print(f"  {len(ds)} items, batch_size=2")

    print("\n=== Running 2 training steps ===\n")
    run_training(
        model,
        dl,
        num_steps=2,
        lr=1e-4,
        log_path="smoke_avqa_log.jsonl",
        ckpt_dir="smoke_avqa_ckpts",
        save_every=0,
    )

    import json
    with open("smoke_avqa_log.jsonl") as f:
        recs = [json.loads(line) for line in f]
    print(f"\nWrote {len(recs)} log records.")
    for r in recs:
        print(f"  step {r.get('step', '?')}: loss={r.get('loss', '?'):.4f}")
    print("\nIf both losses are finite and not NaN, step 3 is done.")


if __name__ == "__main__":
    main()
