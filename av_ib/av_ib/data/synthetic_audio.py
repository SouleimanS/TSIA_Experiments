"""Synthetic audio-signal dataset for architecture validation.

Each example has:
    videos:     random tensor (8, 3, 224, 224)            -- noise, no info
    audio_mels: random tensor (8, 1, 128, 204)
                + planted signal in clip 0 for "Yes" examples only
    prompt:     'Is the signal present in the audio?'
    answer:     'Yes.' or 'No.' based on whether signal was planted

The signal is a constant offset added to all mel bins of clip 0. Strong
enough that ImageBind cannot ignore it, simple enough that the model
should learn the association in <100 steps if the architecture is wired
correctly.

If a model trained on this dataset achieves >70% held-out accuracy, the
architecture can route information from audio input -> Audio Q-Former ->
LLM (via LoRA) -> answer position. If it stays near 50%, there's a bug.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticAudioDataset(Dataset):
    """Deterministic synthetic dataset with planted audio signal."""

    SIGNAL_INTENSITY: float = 5.0
    PROMPT: str = "Is the signal present in the audio?"

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
            audio_mels[0] += self.SIGNAL_INTENSITY

        return {
            "videos": videos,
            "audio_mels": audio_mels,
            "prompt": self.PROMPT,
            "answer": "Yes." if is_yes else "No.",
        }
