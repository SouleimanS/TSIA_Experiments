"""Synthetic video-signal dataset for architecture validation.

Mirror of synthetic_audio.py but the planted signal lives in the video
tensor. Tests the video pipeline: EVA-ViT -> inner Q-Former (12 layers,
BLIP-2 init, 81 missing keys) -> frame_pos_embedding -> outer Q-Former
(2 layers, random init) -> LLM -> answer.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticVideoDataset(Dataset):
    SIGNAL_INTENSITY: float = 5.0
    PROMPT: str = "Is the signal present in the video?"

    def __init__(self, n: int = 64, seed: int = 0, hold_out: bool = False):
        self.n = n
        self.seed_offset = seed + (10_000 if hold_out else 0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(self.seed_offset + idx)
        videos = torch.randn(8, 3, 224, 224, generator=g, dtype=torch.float16)
        audio_mels = torch.randn(8, 1, 128, 204, generator=g)
        is_yes = (idx % 2 == 0)
        if is_yes:
            videos[0] += self.SIGNAL_INTENSITY
        return {
            "videos": videos,
            "audio_mels": audio_mels,
            "prompt": self.PROMPT,
            "answer": "Yes." if is_yes else "No.",
        }
