"""MUSIC-AVQA with on-the-fly noise injection for the robustness experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from av_ib.data.musicavqa import MusicAVQADataset


class MusicAVQADatasetNoisy(Dataset):
    """MUSIC-AVQA with noise injection. Same return dict as MusicAVQADataset."""

    def __init__(
        self,
        ann_path: str | Path,
        video_root: str | Path,
        noise_mode: str,
        noise_sigma: float,
        n_frames: int = 8,
        img_size: int = 224,
        skip_deleted: bool = True,
        seed: int = 42,
    ):
        assert noise_mode in ("gaussian", "audio_mix"), \
            f"noise_mode must be 'gaussian' or 'audio_mix', got {noise_mode!r}"
        assert 0.0 <= noise_sigma <= 2.0, \
            f"noise_sigma must be in [0, 2], got {noise_sigma}"

        self.base = MusicAVQADataset(
            ann_path=ann_path, video_root=video_root,
            n_frames=n_frames, img_size=img_size, skip_deleted=skip_deleted,
        )
        self.noise_mode = noise_mode
        self.noise_sigma = noise_sigma
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def _gen(self, idx: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.seed * 100003 + idx)
        return g

    def _add_gaussian(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        if self.noise_sigma == 0.0:
            return x
        scale = float(x.float().std().item())
        if scale < 1e-6:
            scale = 1.0
        noise = torch.randn(x.shape, generator=gen, dtype=torch.float32) * (self.noise_sigma * scale)
        return (x.float() + noise).to(x.dtype)

    def _audio_mix(self, audio_mel: torch.Tensor, idx: int, gen: torch.Generator) -> torch.Tensor:
        if self.noise_sigma == 0.0:
            return audio_mel
        n = len(self.base)
        other_idx = int(torch.randint(0, n, (1,), generator=gen).item())
        if other_idx == idx:
            other_idx = (other_idx + 1) % n
        try:
            other = self.base[other_idx]
            other_mel = other["audio_mels"]
            if other_mel.shape != audio_mel.shape:
                return audio_mel
            mixed = (1.0 - self.noise_sigma) * audio_mel + self.noise_sigma * other_mel
            return mixed.to(audio_mel.dtype)
        except Exception:
            return audio_mel

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        gen = self._gen(idx)

        if self.noise_mode == "gaussian":
            item["videos"] = self._add_gaussian(item["videos"], gen)
            gen_a = torch.Generator()
            gen_a.manual_seed(self.seed * 100003 + idx + 7919)
            item["audio_mels"] = self._add_gaussian(item["audio_mels"], gen_a)
        elif self.noise_mode == "audio_mix":
            item["audio_mels"] = self._audio_mix(item["audio_mels"], idx, gen)

        return item
