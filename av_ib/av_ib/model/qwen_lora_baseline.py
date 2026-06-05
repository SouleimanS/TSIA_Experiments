"""Qwen3-Omni + LoRA baseline.

Pure fine-tuning of the backbone with no VIB, no fusion, no C-MIB components.
The model is a thin wrapper around QwenOmniWrapper that:
  - exposes forward_train / forward_generate with the same signature as AVModelV6
  - returns a 6-tuple from forward_train (zeros for KL and aux terms) so the
    training loop in av_ib.train.loop is compatible without modification

Two training modes:
    untrained  -- model is loaded but not fine-tuned (use_lora=False, eval only)
    trained    -- model is fine-tuned with LoRA (use_lora=True, default)

The "Qwen3-Omni baseline (untrained)" is just this class instantiated with
use_lora=False and never trained — used as the ceiling comparison.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, Tensor

from av_ib.backbone.qwen_omni import QwenOmniWrapper


class QwenLoRABaseline(nn.Module):
    """Qwen3-Omni with optional LoRA, no bottleneck or fusion.

    forward_train returns (nll, zero, zero, zero, zero, zero) so it is
    drop-in compatible with av_ib.train.loop.run_training when called with
    beta_v=beta_a=beta_j=aux_weight=0.
    """

    def __init__(
        self,
        qwen_model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.qwen = QwenOmniWrapper(
            model_path=qwen_model_path,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

    def forward_train(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        answers: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Returns (nll, 0, 0, 0, 0, 0) — compatible with loop.run_training."""
        nll = self.qwen.forward_train(videos, audios, prompts, answers)
        zero = torch.zeros((), device=nll.device, dtype=nll.dtype)
        return nll, zero, zero, zero, zero, zero

    @torch.no_grad()
    def forward_generate(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        max_new_tokens: int = 20,
    ) -> List[str]:
        return self.qwen.forward_generate(
            videos, audios, prompts, max_new_tokens=max_new_tokens,
        )
