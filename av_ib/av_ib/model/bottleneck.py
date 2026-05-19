"""Information bottleneck modules.

Three implementations:
    Identity         - no bottleneck, passes tokens through (used by v1, v2)
    VIB              - joint Variational Information Bottleneck on all AV tokens
    PerModalityVIB   - separate VIBs for video and audio token streams

All operate on tensors of shape (B, N, D) where D = LLM hidden size (4096).

The VIB has two linear heads (mu, logvar) of shape (D -> D), samples z ~ N(mu, sigma^2)
via the reparameterization trick during training, and returns the mean during eval.
It also returns the per-batch KL divergence to N(0, I), to be added to the loss as
beta * KL.

Trainable param counts (D=4096):
    VIB:           2 * D*D + 2*D    ~ 33.6M
    PerModalityVIB: 2 * (2 * D*D + 2*D) ~ 67.2M
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn, Tensor


class Identity(nn.Module):
    """No-op bottleneck. Used by v1 and v2."""

    def forward(self, av_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        # Return tokens unchanged + a zero KL term so downstream code can
        # always add beta * KL without checking module type.
        kl = torch.zeros((), device=av_tokens.device, dtype=av_tokens.dtype)
        return av_tokens, kl


class VIB(nn.Module):
    """Joint Variational Information Bottleneck.

    Input:  av_tokens (B, N, D)  -- typically (B, 40, 4096)
    Output: z (B, N, D), kl (scalar)

    During training: z = mu + sigma * eps, eps ~ N(0, 1) per element.
    During eval:     z = mu.
    KL is averaged over (B, N) so it stays scale-invariant to token count.
    """

    def __init__(self, d_model: int = 4096):
        super().__init__()
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)
        # Initialize logvar head to output small variance initially, so the
        # bottleneck behaves near-deterministically until KL pressure pushes it.
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -3.0)  # exp(-3) ~ 0.05 stddev

    def forward(self, av_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        mu = self.fc_mu(av_tokens)
        logvar = self.fc_logvar(av_tokens)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        # KL( N(mu, sigma^2) || N(0, I) ) per element = 0.5 * (mu^2 + sigma^2 - logvar - 1)
        kl_per_elem = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        # Average over all dims (batch, token, channel) for scale invariance.
        kl = kl_per_elem.mean()
        return z, kl


class PerModalityVIB(nn.Module):
    """Two VIBs, one for the video token slice and one for audio token slice.

    Args:
        num_video_tokens: how many of the first N tokens are video (default 32).
        d_model: hidden dim (default 4096).

    The split point is fixed at construction. KL is the average of the two.
    """

    def __init__(self, num_video_tokens: int = 32, d_model: int = 4096):
        super().__init__()
        self.num_video_tokens = num_video_tokens
        self.video_vib = VIB(d_model)
        self.audio_vib = VIB(d_model)

    def forward(self, av_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        v = av_tokens[:, : self.num_video_tokens]
        a = av_tokens[:, self.num_video_tokens :]
        zv, kl_v = self.video_vib(v)
        za, kl_a = self.audio_vib(a)
        z = torch.cat([zv, za], dim=1)
        kl = 0.5 * (kl_v + kl_a)
        return z, kl