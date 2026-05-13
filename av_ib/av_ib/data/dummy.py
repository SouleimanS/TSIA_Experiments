"""Dummy AV dataset for training smoke tests.

Returns random tensors of the correct shapes plus made-up text prompts
and answers. The point is to exercise the training loop end-to-end:
collation, optimizer step, gradient flow, JSONL logging. Once real data
is available we'll swap this out for an AVHBench/AVQA loader with the
same {videos, audio_mels, prompts, answers} keys, and nothing else has
to change.

Output shapes (matches what AVModelV1 expects):
    videos:     (T=8, 3, 224, 224)   fp16
    audio_mels: (8 clips, 1, 128, 204)  fp32
    prompts:    str   (single Yes/No-style question)
    answers:    str   (single token answer)
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummyAVDataset(Dataset):
    """Each example is generated deterministically from its index."""

    def __init__(self, n: int = 16, seed: int = 0):
        self.n = n
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        # Per-example RNG so dataloader workers produce the same data.
        g = torch.Generator().manual_seed(self.seed + idx)
        videos = torch.randn(8, 3, 224, 224, generator=g, dtype=torch.float16)
        audio_mels = torch.randn(8, 1, 128, 204, generator=g)
        # Alternate Yes/No so the loss has variation to drive learning,
        # even though the inputs are random.
        prompts = ["Is the dog visible in the video?"]
        answers = ["Yes." if idx % 2 == 0 else "No."]
        return {
            "videos": videos,
            "audio_mels": audio_mels,
            "prompt": prompts[0],
            "answer": answers[0],
        }


def av_collate(batch):
    """Stack videos and audio mels; keep prompts/answers as lists of strings."""
    return {
        "videos": torch.stack([b["videos"] for b in batch], dim=0),
        "audio_mels": torch.stack([b["audio_mels"] for b in batch], dim=0),
        "prompts": [b["prompt"] for b in batch],
        "answers": [b["answer"] for b in batch],
    }
