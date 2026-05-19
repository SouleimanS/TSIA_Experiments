"""Variant 2: Fusion only, no VIB.

Identical to v1 except a MutualCrossAttention block sits between the
Q-Formers and the LLM, letting video and audio tokens attend to each
other before reaching Vicuna.

Trainable in v2:
    - VideoQFormer (~240M)        same as v1
    - AudioQFormer (~54M)         same as v1
    - LoRA on Vicuna (~17M)       same as v1
    - MutualCrossAttention (~50-100M)  NEW
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn, Tensor

from av_ib.model.encoders import VideoEncoder, AudioEncoder
from av_ib.model.qformer import VideoQFormer, AudioQFormer
from av_ib.model.llm import LLMWrapper
from av_ib.model.fusion import MutualCrossAttention


class AVModelV2(nn.Module):
    """Fusion-only baseline. Adds MutualCrossAttention between Q-Formers and LLM."""

    def __init__(
        self,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        num_video_query_tokens: int = 32,
        num_audio_query_tokens: int = 8,
        num_frames: int = 8,
        num_audio_clips: int = 8,
        fusion_heads: int = 8,
        fusion_ffn_mult: int = 1,
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

        # NEW: Fusion module
        self.fusion = MutualCrossAttention(
            d_model=LLMWrapper.HIDDEN_SIZE,
            n_heads=fusion_heads,
            ffn_mult=fusion_ffn_mult,
        )

        # LLM (same as v1)
        self.llm = LLMWrapper(
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

    def _av_tokens(self, videos: Tensor, audio_mels: Tensor) -> Tensor:
        eva_feats = self.video_encoder(videos)
        ib_feats = self.audio_encoder.forward_from_mel(audio_mels)

        video_tokens = self.video_qformer(eva_feats)
        audio_tokens = self.audio_qformer(ib_feats)

        video_tokens, audio_tokens = self.fusion(video_tokens, audio_tokens)

        return torch.cat([video_tokens, audio_tokens], dim=1)

    def forward_train(
        self,
        videos: Tensor,
        audio_mels: Tensor,
        prompts: List[str],
        answers: List[str],
    ) -> Tensor:
        av = self._av_tokens(videos, audio_mels)
        return self.llm.forward_train(av, prompts, answers)

    @torch.no_grad()
    def forward_generate(
        self,
        videos: Tensor,
        audio_mels: Tensor,
        prompts: List[str],
        max_new_tokens: int = 10,
    ) -> List[str]:
        av = self._av_tokens(videos, audio_mels)
        return self.llm.forward_generate(av, prompts, max_new_tokens=max_new_tokens)

    def trainable_summary(self) -> dict:
        out = {}
        for name, child in self.named_children():
            n = sum(p.numel() for p in child.parameters() if p.requires_grad)
            out[name] = n
        out["__total_trainable__"] = sum(out.values())
        out["__total_params__"] = sum(p.numel() for p in self.parameters())
        return out