"""Training loss. Equation (7) of the whitepaper:

    L = -log p_phi(Y | T) + beta * E[ KL(q_theta(T | Zfused) || N(0, I)) ]

The first term is computed inside the LLM wrapper as cross-entropy and
arrives as `output.loss_nll`. The second term is the per-example KL
tensor returned by the bottleneck; we average it across the batch to
get a scalar that matches the units of loss_nll.

For Identity bottlenecks, KL is exactly zero and beta has no effect.
For VIB / PerModalityVIB, beta is the directive's only new hyperparameter.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from av_ib.models.av_model import AVForwardOutput


@dataclass
class LossBreakdown:
    total: Tensor
    nll: Tensor
    kl: Tensor                       # mean over batch
    beta: float


def compute_loss(out: AVForwardOutput, beta: float) -> LossBreakdown:
    kl_mean = out.kl.mean()
    total = out.loss_nll + beta * kl_mean
    return LossBreakdown(total=total, nll=out.loss_nll.detach(), kl=kl_mean.detach(), beta=beta)
