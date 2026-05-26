"""Variant 5 (Qwen3-Omni edition): C-MIB with explicit fusion between per-modality
and joint bottlenecks.

Pipeline:
    audio_enc (B, N_a, 2048) -> VIB_a -> z_a --\
                                                +-> MutualCrossAttn -> [z_v', z_a'] -> concat -> VIB_joint -> Thinker
    video_enc (B, N_v, 2048) -> VIB_v -> z_v --/

Differs from v4 only in adding MutualCrossAttention between the per-modality VIBs and
the joint VIB. v4 concatenates the two streams directly with no fusion mechanism
beyond what the joint VIB's linear maps can do.

The aux heads see PRE-fusion z_v / z_a so each per-modality VIB receives direct
task-supervision (not just gradient via fusion).

Six-term forward_train return: nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j
Trainer composes:
    loss = nll + beta_v * kl_v + beta_a * kl_a + beta_j * kl_j
                + aux_weight * (nll_aux_v + nll_aux_a)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from av_ib.backbone.qwen_omni import QwenOmniWrapper
from av_ib.model.bottleneck import VIB
from av_ib.model.fusion import MutualCrossAttention


class AVModelV5(nn.Module):
    """C-MIB with mutual cross-attention fusion between per-modality and joint VIBs."""

    D_MODEL: int = 2048

    def __init__(
        self,
        qwen_model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        kl_reduction: str = "mean_per_dim",
        fusion_heads: int = 8,
        fusion_ffn_mult: int = 1,
        fusion_n_blocks: int = 1,
    ):
        super().__init__()

        self.qwen = QwenOmniWrapper(
            model_path=qwen_model_path,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        self.bottleneck_v = VIB(d_model=self.D_MODEL, kl_reduction=kl_reduction)
        self.bottleneck_a = VIB(d_model=self.D_MODEL, kl_reduction=kl_reduction)

        self.fusion = MutualCrossAttention(
            d_model=self.D_MODEL,
            n_heads=fusion_heads,
            ffn_mult=fusion_ffn_mult,
            n_blocks=fusion_n_blocks,
        )

        self.bottleneck_joint = VIB(d_model=self.D_MODEL, kl_reduction=kl_reduction)

        self.vocab_size = self.qwen.tokenizer.vocab_size
        self.aux_head_v = nn.Linear(self.D_MODEL, self.vocab_size, bias=False)
        self.aux_head_a = nn.Linear(self.D_MODEL, self.vocab_size, bias=False)

    def _make_provider(self):
        kls, zs = {}, {}

        def provider(audio_out: Tensor, video_out: Tensor) -> Tensor:
            # Move all added modules to the same device AND dtype as the encoder
            # features. Qwen3-Omni uses device_map="auto" + bf16 via accelerate.
            # Our VIBs/fusion/aux heads are added post-hoc and start as fp32 on CPU.
            # First call lazily migrates them to match. We check both, since a
            # module could have right device but wrong dtype (or vice versa).
            target_device = video_out.device
            target_dtype = video_out.dtype
            sample_param = next(self.bottleneck_v.parameters())
            needs_move = (sample_param.device != target_device or
                          sample_param.dtype != target_dtype)
            if needs_move:
                for mod in (self.bottleneck_v, self.bottleneck_a, self.fusion,
                            self.bottleneck_joint, self.aux_head_v, self.aux_head_a):
                    mod.to(device=target_device, dtype=target_dtype)
            z_v, kl_v = self.bottleneck_v(video_out)
            z_a, kl_a = self.bottleneck_a(audio_out)
            z_v_fused, z_a_fused = self.fusion(z_v, z_a)
            av = torch.cat([z_v_fused, z_a_fused], dim=1)
            z_joint, kl_j = self.bottleneck_joint(av)
            kls["v"], kls["a"], kls["j"] = kl_v, kl_a, kl_j
            zs["v"], zs["a"] = z_v, z_a
            return z_joint

        return provider, kls, zs

    def _first_answer_token_ids(self, answers: List[str], device) -> Tensor:
        ids = []
        for ans in answers:
            t = self.qwen.tokenizer(ans, add_special_tokens=False, return_tensors="pt").input_ids
            ids.append(int(t[0, 0].item()) if t.numel() else self.qwen.tokenizer.pad_token_id)
        return torch.tensor(ids, device=device, dtype=torch.long)

    def forward_train(
        self,
        videos: Tensor,
        audios: Tensor,
        prompts: List[str],
        answers: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        provider, kls, zs = self._make_provider()
        nll = self.qwen.forward_train(
            videos, audios, prompts, answers, av_token_provider=provider,
        )
        kl_v, kl_a, kl_j = kls["v"], kls["a"], kls["j"]
        z_v, z_a = zs["v"], zs["a"]
        target = self._first_answer_token_ids(answers, device=nll.device)
        # Aux heads end up on whichever device accelerate placed the encoder
        # features on (via lazy migration in _make_provider). Move target to
        # match each head's output device, since cross_entropy requires it.
        logits_v = self.aux_head_v(z_v.mean(dim=1))
        logits_a = self.aux_head_a(z_a.mean(dim=1))
        nll_aux_v = F.cross_entropy(logits_v, target.to(logits_v.device))
        nll_aux_a = F.cross_entropy(logits_a, target.to(logits_a.device))
        return nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j

    @torch.no_grad()
    def forward_generate(
        self,
        videos: Tensor,
        audios: Tensor,
        prompts: List[str],
        max_new_tokens: int = 10,
    ) -> List[str]:
        provider, _, _ = self._make_provider()
        return self.qwen.forward_generate(
            videos, audios, prompts,
            max_new_tokens=max_new_tokens, av_token_provider=provider,
        )

    def trainable_summary(self) -> dict:
        out = {}
        for name, child in self.named_children():
            n = sum(p.numel() for p in child.parameters() if p.requires_grad)
            out[name] = n
        out["__total_trainable__"] = sum(out.values())
        out["__total_params__"] = sum(p.numel() for p in self.parameters())
        return out
