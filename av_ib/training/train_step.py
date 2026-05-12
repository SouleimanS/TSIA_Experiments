"""A single training step. Returns a dict of scalars for logging.

Kept separate from the smoke-test entry point so the real training loop
can reuse it without modification.
"""
from __future__ import annotations

import torch
from torch import nn

from av_ib.models.av_model import AVModel
from av_ib.training.loss import compute_loss


def train_step(
    model: AVModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: torch.device,
) -> dict:
    model.train()
    x_v = batch["video"].to(device, non_blocking=True)
    x_a = batch["audio"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    out = model(x_v, x_a, labels=labels)
    loss = compute_loss(out, beta=beta)

    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    # Gradient-norm log: useful for catching NaNs and for verifying that
    # only the intended modules receive gradient.
    grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": loss.total.item(),
        "nll": loss.nll.item(),
        "kl": loss.kl.item(),
        "beta": beta,
        "grad_norm": float(grad_norm),
    }
