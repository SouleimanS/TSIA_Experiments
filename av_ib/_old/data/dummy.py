"""Synthetic batches for smoke testing.

A 'batch' here is a dict with keys:
    video:  (B, T_v, H, W, 3)   raw video tensor    -- random
    audio:  (B, L_a)             raw waveform        -- random
    labels: (B, Ly)              answer token ids    -- random in [0, vocab)

Shapes are kept tiny so a forward+backward fits on CPU. The real data
loader will replace this; same dict keys, real tensors.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from av_ib.config import DataCfg


class DummyAVDataset(Dataset):
    """Random AV examples. Length is configurable; defaults small for smoke test."""

    def __init__(self, cfg: DataCfg, n_examples: int = 16, seed: int = 0):
        self.cfg = cfg
        self.n = n_examples
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(self.seed + idx)
        # Keep dims tiny: 4 frames, 16x16 video, 1024-sample audio.
        # The stub encoders use only the channel mean / chunked waveform,
        # so resolution does not matter for the smoke test.
        video = torch.randn(4, 16, 16, 3, generator=g)
        audio = torch.randn(1024, generator=g)
        labels = torch.randint(0, self.cfg.vocab_size, (self.cfg.answer_len,), generator=g)
        return {"video": video, "audio": audio, "labels": labels}


def build_dummy_loader(cfg: DataCfg, batch_size: int, n_examples: int = 16):
    """Build a DataLoader over the dummy dataset. Uses default collation,
    which just stacks tensors of identical shape into a batched tensor."""
    from torch.utils.data import DataLoader
    ds = DummyAVDataset(cfg, n_examples=n_examples)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
