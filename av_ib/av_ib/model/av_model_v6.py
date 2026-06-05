from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from av_ib.backbone.qwen_omni import QwenOmniWrapper
from av_ib.model.bottleneck import VIB
from av_ib.model.fusion import MutualCrossAttention, SinkSymmetricFusion
from av_ib.model.av_model_v6 import SinkAwareVIB, NormTopKSinkVIB


DSINK_DIMS = [985, 1992]
SINK_TAU = 18.0
BETA_SINK_RATIO = 0.01


class AVModelV6(nn.Module):
    """Sink-aware multimodal VIB system (clean audio switch version)."""

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
        sink_dims: List[int] = None,
        sink_tau: float = SINK_TAU,
        beta_sink_ratio: float = BETA_SINK_RATIO,
        variant: str = "a",
        video_vib: str = "sink",
        audio_vib: str = "standard",  # "standard" | "topk"
        adaptive_beta: bool = False,
        adaptive_beta_base: float = 0.1,
        fusion_type: str = "mutual",
    ):
        super().__init__()

        assert variant in ("a", "b", "b_bis", "b_bis_2", "b_bis_3", "b_bis_4", "c")
        assert video_vib in ("sink", "standard")
        assert audio_vib in ("standard", "topk")
        assert fusion_type in ("mutual", "sink_sym", "none")

        self.variant = variant
        self.video_vib = video_vib
        self.audio_vib = audio_vib
        self.adaptive_beta = adaptive_beta
        self.adaptive_beta_base = adaptive_beta_base
        self.fusion_type = fusion_type

        # =========================
        # AUDIO VIB (CLEAN)
        # =========================
        if audio_vib == "standard":
            self.bottleneck_a = VIB(
                d_model=self.D_MODEL,
                kl_reduction=kl_reduction,
            )

        elif audio_vib == "topk":
            self.bottleneck_a = NormTopKSinkVIB(
                d_model=self.D_MODEL,
                top_k_frac=0.40,
                beta_sink_ratio=beta_sink_ratio,
            )

        else:
            raise ValueError(f"audio_vib must be 'standard' or 'topk', got {audio_vib}")

        # =========================
        # QWEN BACKBONE
        # =========================
        self.qwen = QwenOmniWrapper(
            model_path=qwen_model_path,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # =========================
        # VIDEO VIB
        # =========================
        if video_vib == "sink":
            self.bottleneck_v = SinkAwareVIB(
                d_model=self.D_MODEL,
                dsink_dims=sink_dims,
                tau=sink_tau,
                beta_sink_ratio=beta_sink_ratio,
            )
        else:
            self.bottleneck_v = VIB(
                d_model=self.D_MODEL,
                kl_reduction=kl_reduction,
            )

        # =========================
        # FUSION
        # =========================
        if fusion_type == "mutual":
            self.fusion = MutualCrossAttention(
                d_model=self.D_MODEL,
                n_heads=fusion_heads,
                ffn_mult=fusion_ffn_mult,
                n_blocks=fusion_n_blocks,
            )
        elif fusion_type == "sink_sym":
            self.fusion = SinkSymmetricFusion(
                d_model=self.D_MODEL,
                n_heads=fusion_heads,
                dsink_dims=sink_dims,
                sink_tau=sink_tau,
            )
        else:
            self.fusion = None

        # =========================
        # JOINT VIB
        # =========================
        if variant == "a":
            self.bottleneck_joint = SinkAwareVIB(
                d_model=self.D_MODEL,
                dsink_dims=sink_dims,
                tau=sink_tau,
                beta_sink_ratio=beta_sink_ratio,
            )
        elif variant in ("b", "b_bis", "b_bis_2", "b_bis_3", "b_bis_4"):
            self.bottleneck_joint = None
        else:
            self.bottleneck_joint = VIB(
                d_model=self.D_MODEL,
                kl_reduction=kl_reduction,
            )

        self.vocab_size = self.qwen.tokenizer.vocab_size
        self.aux_head_v = nn.Linear(self.D_MODEL, self.vocab_size, bias=False)
        self.aux_head_a = nn.Linear(self.D_MODEL, self.vocab_size, bias=False)

    # =========================
    # FORWARD CORE
    # =========================
    def _make_provider(self):
        kls, zs, masks = {}, {}, {}

        def provider(audio_out: Tensor, video_out: Tensor) -> Tensor:
            # --- VIDEO ---
            if self.video_vib == "sink":
                z_v, kl_v, mask_v = self.bottleneck_v(video_out)
                masks["sink_v"] = mask_v
            else:
                z_v, kl_v = self.bottleneck_v(video_out)
                masks["sink_v"] = None

            # --- AUDIO ---
            out_a = self.bottleneck_a(audio_out)
            z_a = out_a[0]
            kl_a = out_a[1]

            # --- FUSION ---
            if self.fusion is not None:
                z_v, z_a = self.fusion(z_v, z_a)

            av = torch.cat([z_v, z_a], dim=1)

            # --- JOINT ---
            if self.variant == "a":
                z_j, kl_j, mask_j = self.bottleneck_joint(av)
                masks["sink_j"] = mask_j

            elif self.variant in ("b", "b_bis", "b_bis_2", "b_bis_3", "b_bis_4"):
                z_j = av
                kl_j = torch.zeros((), device=av.device, dtype=av.dtype)
                masks["sink_j"] = None

            else:
                z_j, kl_j = self.bottleneck_joint(av)
                masks["sink_j"] = None

            kls["v"], kls["a"], kls["j"] = kl_v, kl_a, kl_j
            zs["v"], zs["a"] = z_v, z_a

            return z_j

        return provider, kls, zs, masks

    # =========================
    # TRAIN STEP
    # =========================
    def forward_train(self, videos, audios, prompts, answers):
        provider, kls, zs, _ = self._make_provider()

        nll = self.qwen.forward_train(
            videos, audios, prompts, answers,
            av_token_provider=provider,
        )

        kl_v, kl_a, kl_j = kls["v"], kls["a"], kls["j"]

        z_v, z_a = zs["v"], zs["a"]

        target = torch.zeros_like(nll)  # placeholder-safe (unchanged behavior assumed)

        logits_v = self.aux_head_v(z_v.mean(dim=1))
        logits_a = self.aux_head_a(z_a.mean(dim=1))

        nll_aux_v = F.cross_entropy(logits_v, target.to(logits_v.device))
        nll_aux_a = F.cross_entropy(logits_a, target.to(logits_a.device))

        return nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j