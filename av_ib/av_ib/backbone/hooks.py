"""Forward hooks on Qwen3-Omni encoders to capture (and optionally replace) outputs."""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn, Tensor


class EncoderTap:
    """Registers forward hooks on the audio and visual encoders."""

    def __init__(self, audio_encoder: nn.Module, visual_encoder: nn.Module):
        self.audio_out: Optional[Tensor] = None
        self.video_out: Optional[Tensor] = None
        self.provider: Optional[Callable] = None
        self._h_audio = audio_encoder.register_forward_hook(self._capture_audio)
        self._h_video = visual_encoder.register_forward_hook(self._capture_video)

    def _capture_audio(self, module, inputs, output):
        self.audio_out = _extract_tensor(output)
        return None

    def _capture_video(self, module, inputs, output):
        self.video_out = _extract_tensor(output)
        return None

    def set_provider(self, provider: Callable):
        self.provider = provider

    def remove(self):
        self._h_audio.remove()
        self._h_video.remove()


def _extract_tensor(output):
    if torch.is_tensor(output):
        return output
    for attr in ("last_hidden_state", "hidden_states", "features"):
        if hasattr(output, attr):
            return getattr(output, attr)
    raise TypeError(f"Cannot extract tensor from {type(output).__name__}")
