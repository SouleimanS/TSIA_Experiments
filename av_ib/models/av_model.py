"""End-to-end AV model assembly.

Pipeline:

    Xv ----video encoder---> Ev ----video projector----> Zv ---\
                                                                 fusion -> Zfused -> bottleneck -> T -> LLM -> answer
    Xa ----audio encoder---> Ea ----audio projector----> Za ---/

Variant routing (corresponds to the six ablations the user listed):

    v1 no-fusion / no-VIB:   fusion=identity,            bottleneck=identity
    v2 cross-attention only: fusion=mutual_cross_attn,   bottleneck=identity
    v3 VIB only:             fusion=identity,            bottleneck=vib
    v4 Φ + VIB (the proposal): fusion=mutual_cross_attn, bottleneck=vib
    v5 separate VIBs only:   fusion=identity,            bottleneck=per_modality_vib
    v6 separate VIBs + fused VIB: bottleneck=per_modality_vib  AND a post-fusion VIB.

The 'per_modality_vib' bottleneck lives *before* fusion in the forward
graph, not after it. To keep the config schema uniform, when
`bottleneck.name == 'per_modality_vib'` we route Zv, Za through it
before fusion and place an IdentityBottleneck (or another VIB for v6)
after fusion. Variant v6 is handled by setting both `bottleneck.name =
'per_modality_vib'` and `bottleneck.kwargs.add_fused_vib = True`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor

from av_ib.config import RunCfg
from av_ib.models.encoders import (
    build_video_encoder, build_audio_encoder, build_projector, freeze,
)
from av_ib.models.fusion import build_fusion
from av_ib.models.bottleneck import build_bottleneck, VIB, PerModalityVIB, IdentityBottleneck
from av_ib.models.llm import build_llm


@dataclass
class AVForwardOutput:
    logits: Tensor
    loss_nll: Tensor              # cross-entropy from LLM
    kl: Tensor                    # (B,) KL contribution; zero for identity bottlenecks
    T: Tensor                     # bottlenecked tokens (B, N, d) for diagnostics
    Zfused: Tensor                # for MI-measurement hooks


class AVModel(nn.Module):
    def __init__(self, cfg: RunCfg):
        super().__init__()
        self.cfg = cfg
        m = cfg.model

        # 1. Encoders. Frozen by convention; the smoke-test stubs are tiny
        #    and trainable but we freeze them so the only trainable modules
        #    are exactly what the directive specifies (Φ and qθ).
        self.video_encoder = build_video_encoder(m.video_encoder.name, **m.video_encoder.kwargs)
        self.audio_encoder = build_audio_encoder(m.audio_encoder.name, **m.audio_encoder.kwargs)
        freeze(self.video_encoder)
        freeze(self.audio_encoder)

        # 2. Projectors. Also frozen.
        self.video_projector = build_projector(
            m.video_projector.name,
            d_in=m.video_projector.d_in, d_out=m.video_projector.d_out,
            num_tokens=m.video_projector.num_tokens, **m.video_projector.kwargs,
        )
        self.audio_projector = build_projector(
            m.audio_projector.name,
            d_in=m.audio_projector.d_in, d_out=m.audio_projector.d_out,
            num_tokens=m.audio_projector.num_tokens, **m.audio_projector.kwargs,
        )
        freeze(self.video_projector)
        freeze(self.audio_projector)

        d = m.llm.hidden_size
        assert m.video_projector.d_out == d and m.audio_projector.d_out == d, (
            f"projector outputs ({m.video_projector.d_out}, {m.audio_projector.d_out}) "
            f"must match LLM hidden ({d})"
        )

        # 3. Bottleneck placement: per-modality vs post-fusion.
        bn_name = m.bottleneck.name
        bn_kw = dict(m.bottleneck.kwargs)
        self._pre_fusion_bottleneck = None
        self._add_fused_vib_for_v6 = False

        if bn_name == "per_modality_vib":
            add_fused_vib = bn_kw.pop("add_fused_vib", False)
            self._pre_fusion_bottleneck = build_bottleneck("per_modality_vib", d=d, **bn_kw)
            # Variant v6: also keep a VIB after fusion.
            if add_fused_vib:
                self.bottleneck = VIB(d=d, **bn_kw)
                self._add_fused_vib_for_v6 = True
            else:
                self.bottleneck = IdentityBottleneck()
        else:
            self.bottleneck = build_bottleneck(bn_name, d=d, **bn_kw)

        # 4. Fusion (trained).
        self.fusion = build_fusion(m.fusion.name, d=d, **m.fusion.kwargs)

        # 5. LLM (frozen).
        self.llm = build_llm(m.llm.name, hidden_size=d, **m.llm.kwargs)
        freeze(self.llm)

    # --------------------------------------------------------------------
    # Introspection helpers
    # --------------------------------------------------------------------

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_module_summary(self) -> dict[str, int]:
        """For each named child, count parameters with requires_grad=True.
        Useful to confirm 'only Φ and qθ are trained' before launching."""
        out = {}
        for name, mod in self.named_children():
            n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            out[name] = n
        if self._pre_fusion_bottleneck is not None:
            n = sum(p.numel() for p in self._pre_fusion_bottleneck.parameters() if p.requires_grad)
            out["_pre_fusion_bottleneck"] = n
        return out

    # --------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------

    def forward(self, x_v: Tensor, x_a: Tensor, labels: Tensor | None = None) -> AVForwardOutput:
        # Encoders + projectors (frozen).
        with torch.no_grad():
            Ev = self.video_encoder(x_v)
            Ea = self.audio_encoder(x_a)
            Zv = self.video_projector(Ev)
            Za = self.audio_projector(Ea)
        # Detach to make the frozen boundary explicit -- prevents anything
        # downstream from accidentally trying to backprop into the encoders
        # if they are ever unfrozen by mistake.
        Zv = Zv.detach()
        Za = Za.detach()

        # Pre-fusion bottleneck path (variants v5, v6).
        kl_pre = torch.zeros(Zv.size(0), device=Zv.device, dtype=Zv.dtype)
        if self._pre_fusion_bottleneck is not None:
            Zv, Za, kl_pre = self._pre_fusion_bottleneck(Zv, Za)

        # Fusion.
        Zfused, _ = self.fusion(Zv, Za)

        # Post-fusion bottleneck (variants v3, v4, v6).
        T, kl_post = self.bottleneck(Zfused)

        kl = kl_pre + kl_post

        # LLM.
        out = self.llm(inputs_embeds=T, labels=labels)
        return AVForwardOutput(
            logits=out.logits,
            loss_nll=out.loss if out.loss is not None else torch.tensor(0.0, device=T.device),
            kl=kl,
            T=T,
            Zfused=Zfused,
        )
