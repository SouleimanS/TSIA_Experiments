"""AVModelBalanced: AVModelV6 with two audio-attention fixes.

Fix 1 — token-count equalization (audio_balance=True):
    After the VIB, multiply audio tokens by sqrt(n_v / n_a).
    This makes audio's total attention budget equal to video's,
    correcting the structural 17:1 count imbalance.

Fix 2 — direction-preservation loss (dir_weight > 0):
    Add lambda * (1 - cos_sim(z_a, audio_raw)).mean() to the training loss.
    This penalises the VIB for rotating audio tokens away from their original
    direction, which is the root cause of the key-norm suppression (cos_sim
    dropped to 0.24 in token_analysis, meaning large rotation into a subspace
    that k_proj cannot project onto strongly).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from av_ib.model.av_model_v6 import AVModelV6


class AVModelBalanced(AVModelV6):

    def __init__(
        self,
        *args,
        audio_balance: bool = True,
        dir_weight: float = 0.1,
        **kwargs,
    ):
        """All AVModelV6 kwargs accepted. Extra:

        audio_balance : if True, scale audio tokens by sqrt(n_v/n_a) in the
                        provider so their total attention budget equals video's.
        dir_weight    : weight for the direction-preservation loss on z_a.
                        Set to 0.0 to disable (option 1 only).
        """
        super().__init__(*args, **kwargs)
        self.audio_balance = audio_balance
        self.dir_weight = dir_weight
        self._last_dir_loss: Tensor | None = None

    # ------------------------------------------------------------------
    # Provider override
    # ------------------------------------------------------------------
    def _make_provider(self):
        base_provider, kls, zs, masks = super()._make_provider()
        model = self   # capture for closure

        def balanced_provider(audio_out: Tensor, video_out: Tensor) -> Tensor:
            z_j = base_provider(audio_out, video_out)   # [video | audio] concatenated

            n_v = zs["v"].size(1)
            n_a = zs["a"].size(1)

            # ── Fix 2: direction-preservation loss ───────────────────────
            if model.dir_weight > 0 and model.training:
                z_a = z_j[:, n_v:, :]                      # (1, N_a, D)
                # audio_out is already fp32 (cast inside base provider)
                # We need the raw audio_out before the VIB.
                # It's not stored, but we can approximate via the cosine
                # similarity between z_a and the DETACHED base audio_out.
                # Re-cast audio_out to match z_a dtype/device.
                a_ref = audio_out.to(z_j).detach().float()[:, :n_a, :]
                z_a_f = z_a.float()
                cos = F.cosine_similarity(
                    z_a_f.reshape(-1, z_a_f.size(-1)),
                    a_ref.reshape(-1, a_ref.size(-1)),
                    dim=-1,
                )
                model._last_dir_loss = (1.0 - cos).mean()
            else:
                model._last_dir_loss = None

            # ── Fix 1: token-count equalization ──────────────────────────
            if model.audio_balance and n_v > 0 and n_a > 0:
                scale = math.sqrt(n_v / n_a)
                z_j = torch.cat([
                    z_j[:, :n_v, :],
                    z_j[:, n_v:, :] * scale,
                ], dim=1)

            return z_j

        return balanced_provider, kls, zs, masks

    # ------------------------------------------------------------------
    # Training forward: add direction loss to the returned tuple
    # ------------------------------------------------------------------
    def forward_train(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        answers: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        result = super().forward_train(videos, audios, prompts, answers)
        # result = (nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j)
        if self.dir_weight > 0 and self._last_dir_loss is not None:
            nll = result[0] + self.dir_weight * self._last_dir_loss.to(result[0])
            result = (nll,) + result[1:]
        return result

    def _collect_diagnostics(self) -> dict:
        out = super()._collect_diagnostics()
        if self._last_dir_loss is not None:
            out["dir_loss"] = float(self._last_dir_loss.item())
        return out
