"""Variant 4: C-MIB (Complete Multimodal Information Bottleneck).

Stacks three VIBs:
    - VIB_v: per-modality VIB on video tokens (B, 32, 4096)
    - VIB_a: per-modality VIB on audio tokens (B, 8, 4096)
    - VIB_joint: joint VIB on the concatenated z_v + z_a tokens (B, 40, 4096)

Plus two small auxiliary classification heads that predict the FIRST gold
answer token from pooled z_v and z_a respectively. Following Mai et al. 2023
(C-MIB), this gives each per-modality VIB direct gradient pressure to retain
task-relevant information, rather than relying only on gradient through the
joint path. The aux heads project to the full Vicuna vocab (32000) and apply
cross-entropy against the first token of the gold answer — dataset-agnostic.

forward_train returns six terms:
    nll       : main LLM cross-entropy on the answer span
    nll_aux_v : aux head loss on first answer token, from pooled z_v
    nll_aux_a : aux head loss on first answer token, from pooled z_a
    kl_v      : KL of video VIB
    kl_a      : KL of audio VIB
    kl_j      : KL of joint VIB

The trainer composes:
    loss = nll + beta_v * kl_v + beta_a * kl_a + beta_j * kl_j
                + aux_weight * (nll_aux_v + nll_aux_a)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from av_ib.model.encoders import VideoEncoder, AudioEncoder
from av_ib.model.qformer import VideoQFormer, AudioQFormer
from av_ib.model.llm import LLMWrapper
from av_ib.model.bottleneck import VIB


class AVModelV4(nn.Module):
    """C-MIB: per-modality VIBs + joint VIB + aux heads."""

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
        self.num_video_query_tokens = num_video_query_tokens
        self.num_audio_query_tokens = num_audio_query_tokens

        self.video_encoder = VideoEncoder(precision="fp16")
        self.audio_encoder = AudioEncoder()

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

        self.bottleneck_v = VIB(d_model=LLMWrapper.HIDDEN_SIZE)
        self.bottleneck_a = VIB(d_model=LLMWrapper.HIDDEN_SIZE)
        self.bottleneck_joint = VIB(d_model=LLMWrapper.HIDDEN_SIZE)

        self.llm = LLMWrapper(
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        self.vocab_size = self.llm.tokenizer.vocab_size

        self.aux_head_v = nn.Linear(LLMWrapper.HIDDEN_SIZE, self.vocab_size, bias=False)
        self.aux_head_a = nn.Linear(LLMWrapper.HIDDEN_SIZE, self.vocab_size, bias=False)

    def _av_tokens(
        self, videos: Tensor, audio_mels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        eva_feats = self.video_encoder(videos)
        ib_feats = self.audio_encoder.forward_from_mel(audio_mels)

        video_tokens = self.video_qformer(eva_feats)
        audio_tokens = self.audio_qformer(ib_feats)

        z_v, kl_v = self.bottleneck_v(video_tokens)
        z_a, kl_a = self.bottleneck_a(audio_tokens)

        av = torch.cat([z_v, z_a], dim=1)
        z_joint, kl_j = self.bottleneck_joint(av)

        return z_joint, z_v, z_a, kl_v, kl_a, kl_j

    def _first_answer_token_ids(self, answers: List[str], device) -> Tensor:
        first_ids = []
        for ans in answers:
            ids = self.llm.tokenizer(
                ans, add_special_tokens=False, return_tensors="pt"
            ).input_ids
            if ids.numel() == 0:
                first_ids.append(self.llm.tokenizer.pad_token_id)
            else:
                first_ids.append(int(ids[0, 0].item()))
        return torch.tensor(first_ids, device=device, dtype=torch.long)

    def forward_train(
        self,
        videos: Tensor,
        audio_mels: Tensor,
        prompts: List[str],
        answers: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        z_joint, z_v, z_a, kl_v, kl_a, kl_j = self._av_tokens(videos, audio_mels)

        nll = self.llm.forward_train(z_joint, prompts, answers)

        pooled_v = z_v.mean(dim=1)
        pooled_a = z_a.mean(dim=1)
        logits_v = self.aux_head_v(pooled_v)
        logits_a = self.aux_head_a(pooled_a)

        target = self._first_answer_token_ids(answers, device=z_joint.device)
        nll_aux_v = F.cross_entropy(logits_v, target)
        nll_aux_a = F.cross_entropy(logits_a, target)

        return nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j

    @torch.no_grad()
    def forward_generate(
        self,
        videos: Tensor,
        audio_mels: Tensor,
        prompts: List[str],
        max_new_tokens: int = 10,
    ) -> List[str]:
        z_joint, _, _, _, _, _ = self._av_tokens(videos, audio_mels)
        return self.llm.forward_generate(z_joint, prompts, max_new_tokens=max_new_tokens)

    def trainable_summary(self) -> dict:
        out = {}
        for name, child in self.named_children():
            n = sum(p.numel() for p in child.parameters() if p.requires_grad)
            out[name] = n
        out["__total_trainable__"] = sum(out.values())
        out["__total_params__"] = sum(p.numel() for p in self.parameters())
        return out
