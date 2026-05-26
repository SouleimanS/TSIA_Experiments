"""Variant 3: VIB only, no Fusion.

Identical to v1 except a joint VIB bottleneck sits between the concatenated
AV tokens and the LLM, compressing the (B, 40, 4096) tokens via a stochastic
encoding during training, deterministic mean during eval.

The forward returns a (loss, kl) pair from forward_train, so the training loop
can add beta * KL to the loss.

Trainable in v3:
    - VideoQFormer (~240M)        same as v1
    - AudioQFormer (~54M)         same as v1
    - LoRA on Vicuna (~17M)       same as v1
    - VIB (~33.6M)                NEW
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, Tensor

from av_ib.model.encoders import VideoEncoder, AudioEncoder
from av_ib.model.qformer import VideoQFormer, AudioQFormer
from av_ib.model.llm import LLMWrapper
from av_ib.model.bottleneck import VIB


class AVModelV3(nn.Module):
    """VIB-only baseline. Joint VIB between concatenated AV tokens and LLM."""

    def __init__(
        self,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        num_video_query_tokens: int = 32,
        num_audio_query_tokens: int = 8,
        num_frames: int = 8,
        num_audio_clips: int = 8,
    ):
        super().__init__()

        # Frozen encoders (same as v1)
        self.video_encoder = VideoEncoder(precision="fp16")
        self.audio_encoder = AudioEncoder()

        # Trainable Q-Formers (same as v1)
        self.video_qformer = VideoQFormer(
            num_query_tokens=num_video_query_tokens,
            num_frames=num_frames,
            llm_hidden=LLMWrapper.HIDDEN_SIZE,
        )
        self.audio_qformer = AudioQFormer(
            num_query_tokens=num_audio_query_tokens,
            num_clips=num_audio_clips,
            llm_hidden=LLMWrapper.HIDDEN_SIZE,
        )

        # NEW: VIB bottleneck on concatenated AV tokens
        self.bottleneck = VIB(d_model=LLMWrapper.HIDDEN_SIZE)

        # LLM (same as v1)
        self.llm = LLMWrapper(
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

    def _av_tokens(self, videos: Tensor, audio_mels: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (z, kl). z is (B, 40, 4096), kl is scalar."""
        eva_feats = self.video_encoder(videos)
        ib_feats = self.audio_encoder.forward_from_mel(audio_mels)

        video_tokens = self.video_qformer(eva_feats)
        audio_tokens = self.audio_qformer(ib_feats)

        av = torch.cat([video_tokens, audio_tokens], dim=1)
        z, kl = self.bottleneck(av)
        return z, kl

    def forward_train(
        self,
        videos: Tensor,
        audio_mels: Tensor,
        prompts: List[str],
        answers: List[str],
    ) -> Tuple[Tensor, Tensor]:
        """Returns (nll_loss, kl). Training loop adds beta * kl to nll."""
        z, kl = self._av_tokens(videos, audio_mels)
        nll = self.llm.forward_train(z, prompts, answers)
        return nll, kl

    @torch.no_grad()
    def forward_generate(
        self,
        videos: Tensor,
        audio_mels: Tensor,
        prompts: List[str],
        max_new_tokens: int = 10,
    ) -> List[str]:
        z, _ = self._av_tokens(videos, audio_mels)
        return self.llm.forward_generate(z, prompts, max_new_tokens=max_new_tokens)

    def trainable_summary(self) -> dict:
        out = {}
        for name, child in self.named_children():
            n = sum(p.numel() for p in child.parameters() if p.requires_grad)
            out[name] = n
        out["__total_trainable__"] = sum(out.values())
        out["__total_params__"] = sum(p.numel() for p in self.parameters())
        return out