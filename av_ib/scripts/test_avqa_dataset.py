"""Verify AVQADataset produces tensors of the right shape from real clips."""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from av_ib.data.avqa import AVQADataset
from av_ib.data.dummy import av_collate
from torch.utils.data import DataLoader

ANN = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_train.json"
VID = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVQA" / "videos" / "Train"

print(f"=== Building AVQADataset ===")
ds = AVQADataset(ann_path=ANN, video_root=VID)
print(f"  {len(ds)} records")

print(f"\n=== Single __getitem__ ===")
t0 = time.time()
item = ds[0]
print(f"  loaded in {time.time()-t0:.2f}s")
print(f"  videos:     {tuple(item['videos'].shape)} dtype={item['videos'].dtype}")
print(f"  audio_mels: {tuple(item['audio_mels'].shape)} dtype={item['audio_mels'].dtype}")
print(f"  prompt:     {item['prompt']!r}")
print(f"  answer:     {item['answer']!r}")

print(f"\n=== Batch of 2 (collate) ===")
dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=av_collate, num_workers=0)
batch = next(iter(dl))
print(f"  videos:     {tuple(batch['videos'].shape)} dtype={batch['videos'].dtype}")
print(f"  audio_mels: {tuple(batch['audio_mels'].shape)} dtype={batch['audio_mels'].dtype}")
print(f"  prompts:    {batch['prompts']}")
print(f"  answers:    {batch['answers']}")

print(f"\n=== Compare to dummy contract ===")
print(f"  Dummy expects videos     (B, 8, 3, 224, 224) fp16        -> got {tuple(batch['videos'].shape)} {batch['videos'].dtype}")
print(f"  Dummy expects audio_mels (B, 8, 1, 128, 204) fp32        -> got {tuple(batch['audio_mels'].shape)} {batch['audio_mels'].dtype}")
