"""AVModelGated: AVModelV6 + a prompt-conditioned modality gate.

The gate predicts p = (p_video, p_audio, p_text) from the prompt and scales the
injected video / audio token embeddings by (M * p_modality) before they are
spliced into the LLM's inputs_embeds. M* centres the gate at uniform so an
untrained (zero-init) gate is the identity, and the model reduces exactly to
the ungated baseline.

Everything else — the SinkAwareVIB video bottleneck, the audio bottleneck, the
aux heads, the 6-term forward_train contract, the splicer — is inherited
unchanged from AVModelV6, so this drops straight into the existing training
loop (av_ib.train.loop.run_training).

Batch-size-1 assumption: like the rest of the pipeline (see CMIBSplicer), this
model assumes B=1 per forward. The gate computes one distribution per prompt.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from av_ib.model.av_model_v6 import AVModelV6
from av_ib.modality_gate.gate import ModalityGate, MODALITY_NAMES


class AVModelGated(AVModelV6):

    def __init__(self, *args, gate_hidden: int = 512,
                 gate_centered: bool = True, **kwargs):
        """All AVModelV6 kwargs are accepted. Extra:

        gate_hidden   : hidden width of the gate MLP.
        gate_centered : if True, scale by (M * p) so uniform p == identity.
                        if False, scale by raw p (always shrinks; for ablation).
        """
        super().__init__(*args, **kwargs)
        self.gate = ModalityGate(self.D_MODEL, hidden=gate_hidden,
                                 n_modalities=len(MODALITY_NAMES))
        self.gate_centered = gate_centered
        # Per-forward state (B=1)
        self._current_modality_p: Tensor | None = None
        self.last_modality_p: Tensor | None = None

    # ------------------------------------------------------------------
    # Prompt -> pooled representation -> gate probabilities
    # ------------------------------------------------------------------
    def _prompt_repr(self, prompt: str) -> Tensor:
        """Mean-pooled frozen input embeddings of the prompt tokens. (1, D)."""
        tok = self.qwen.tokenizer
        ids = tok(prompt, add_special_tokens=False,
                  return_tensors="pt").input_ids
        embed = self.qwen.thinker_text_model.get_input_embeddings()
        dev = embed.weight.device
        with torch.no_grad():
            e = embed(ids.to(dev))            # (1, L, D), bf16
        return e.float().mean(dim=1)          # (1, D), fp32

    def _compute_p(self, prompt: str) -> Tensor:
        rep = self._prompt_repr(prompt)
        dev = next(self.gate.parameters()).device
        if dev.type != rep.device.type or dev != rep.device:
            self.gate.to(rep.device)
            dev = rep.device
        return self.gate(rep.to(dev))         # (1, M)

    # ------------------------------------------------------------------
    # Provider with gating applied to the spliced embeddings
    # ------------------------------------------------------------------
    def _make_provider(self):
        base_provider, kls, zs, masks = super()._make_provider()

        def gated_provider(audio_out: Tensor, video_out: Tensor) -> Tensor:
            z_j = base_provider(audio_out, video_out)   # [video | audio]
            p = self._current_modality_p
            if p is None:
                return z_j

            p = p.to(device=z_j.device, dtype=z_j.dtype)
            M = self.gate.n_modalities
            scale = float(M) if self.gate_centered else 1.0
            n_v = zs["v"].size(1)               # video tokens come first

            s_v = scale * p[0, 0]
            s_a = scale * p[0, 1]
            z_j = z_j.clone()
            z_j[:, :n_v, :] = z_j[:, :n_v, :] * s_v
            z_j[:, n_v:, :] = z_j[:, n_v:, :] * s_a

            self.last_modality_p = p.detach().float().cpu()
            return z_j

        return gated_provider, kls, zs, masks

    # ------------------------------------------------------------------
    # Training forward: compute p first, stash, delegate to AVModelV6
    # ------------------------------------------------------------------
    def forward_train(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        answers: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert len(prompts) == 1, "AVModelGated assumes batch_size=1"
        self._current_modality_p = self._compute_p(prompts[0])
        try:
            return super().forward_train(videos, audios, prompts, answers)
        finally:
            self._current_modality_p = None

    # ------------------------------------------------------------------
    # Generation forward: per-prompt gating
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_generate(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        max_new_tokens: int = 20,
    ) -> List[str]:
        outs: List[str] = []
        for v, a, pr in zip(videos, audios, prompts):
            self._current_modality_p = self._compute_p(pr)
            try:
                outs.extend(
                    super().forward_generate([v], [a], [pr],
                                             max_new_tokens=max_new_tokens)
                )
            finally:
                self._current_modality_p = None
        return outs

    # ------------------------------------------------------------------
    # Diagnostics: surface the gate distribution in the training log
    # ------------------------------------------------------------------
    def _collect_diagnostics(self) -> dict:
        out = super()._collect_diagnostics()
        if self.last_modality_p is not None:
            vals = self.last_modality_p.flatten().tolist()
            for name, v in zip(MODALITY_NAMES, vals):
                out[f"p_{name}"] = float(v)
        return out
