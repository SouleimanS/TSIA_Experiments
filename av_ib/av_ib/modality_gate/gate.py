"""ModalityGate: prompt -> distribution over modalities.

A small MLP that maps a pooled prompt representation to a softmax distribution
over {video, audio, text}. Kept in fp32 (like the bottleneck modules) so AdamW
updates at lr~1e-4 are not silently dropped by bf16 ULP rounding.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor


# Default ordering of the modality axis. Index 0/1 are the gated (injected)
# streams; index 2 (text) is computed but currently acts as a "neither" sink
# that lets the gate keep probability mass off video/audio when the prompt is
# self-contained. Keeping it in the softmax makes the distribution proper.
MODALITY_NAMES = ("video", "audio", "text")


class ModalityGate(nn.Module):
    """MLP gate: (B, d_model) prompt representation -> (B, M) probabilities."""

    def __init__(self, d_model: int, hidden: int = 512,
                 n_modalities: int = 3, dropout: float = 0.0):
        super().__init__()
        self.n_modalities = n_modalities
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, n_modalities),
        )
        # Zero-init the final layer so the gate starts at the uniform
        # distribution (logits all 0 -> softmax uniform). Combined with the
        # M* centring in the model, an untrained gate is exactly the identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, prompt_repr: Tensor) -> Tensor:
        """prompt_repr: (B, d_model) float -> (B, n_modalities) probabilities."""
        logits = self.net(prompt_repr.float())
        return torch.softmax(logits, dim=-1)
