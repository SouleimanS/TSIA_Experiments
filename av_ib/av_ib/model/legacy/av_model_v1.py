"""Variant 1: no fusion, no VIB. The minimal end-to-end pipeline.

This is the baseline against which v2-v6 will be compared. Architecture:

    video --> EVA-ViT --> VideoQFormer ----\
                                            concat --> Vicuna-v0 (+ LoRA)
    audio --> ImageBind --> AudioQFormer --/

Variants v2-v6 will add Fusion (Phi) and/or Bottleneck (q_theta) between
the Q-Formers and the LLM. v1 has neither: AV tokens go straight to the
LLM, with the LLM's self-attention being the only mechanism that mixes
modalities (and the Q-Formers being the only mechanism that compresses).

Trainable in v1:
    - VideoQFormer (~140M)
    - AudioQFormer (~30M)
    - LoRA on Vicuna q/k/v/o (~17M)
Frozen:
    - EVA-ViT, ImageBind, Vicuna base weights

Forward signatures match LLMWrapper's:
    forward_train(videos, audios, prompts, answers) -> loss
    forward_generate(videos, audios, prompts) -> List[str]
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn, Tensor

from av_ib.model.encoders import VideoEncoder, AudioEncoder
from av_ib.model.qformer import VideoQFormer, AudioQFormer
from av_ib.model.llm import LLMWrapper


class AVModelV1(nn.Module):
    """No-fusion, no-VIB baseline.

    Constructor flags let us turn things on/off without rebuilding the class.
    For v1, the defaults are correct (no fusion, no bottleneck).
    """

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

        # --- Frozen encoders ---
        self.video_encoder = VideoEncoder(precision="fp16")
        self.audio_encoder = AudioEncoder()

        # --- Trainable Q-Formers ---
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

        # --- LLM ---
        self.llm = LLMWrapper(
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

    # ------------------------------------------------------------------
    # Shared AV path -- produces (B, 40, 4096) tokens for the LLM.
    # ------------------------------------------------------------------

    def _av_tokens(self, videos: Tensor, audio_mels: Tensor) -> Tensor:
        """videos:      (B, T=8, 3, 224, 224)
           audio_mels:  (B, N_clips=8, 1, 128, 204)
           returns:     (B, 40, 4096)
        """
        # Frozen encoder forward (under torch.no_grad inside each class).
        eva_feats = self.video_encoder(videos)                 # (B*T, 257, 1408)
        ib_feats = self.audio_encoder.forward_from_mel(audio_mels)  # (B, 8, 1024)

        # Trainable Q-Formers.
        video_tokens = self.video_qformer(eva_feats)            # (B, 32, 4096)
        audio_tokens = self.audio_qformer(ib_feats)             # (B, 8, 4096)

        # Concatenate along token axis.
        return torch.cat([video_tokens, audio_tokens], dim=1)   # (B, 40, 4096)

    # ------------------------------------------------------------------
    # Training and generation modes
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def trainable_summary(self) -> dict:
        """For each named child, count trainable params. Useful sanity check."""
        out = {}
        for name, child in self.named_children():
            n = sum(p.numel() for p in child.parameters() if p.requires_grad)
            out[name] = n
        out["__total_trainable__"] = sum(out.values())
        out["__total_params__"] = sum(p.numel() for p in self.parameters())
        return out
