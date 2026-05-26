"""Single-GPU training loop for AVModelV5 (Qwen3-Omni + C-MIB).

Replaces the Vicuna-era loop.py. Key differences:
    - forward_train returns 6 losses (nll, nll_aux_v/a, kl_v/a/j) — composed here
    - Inputs are file paths (str), not pre-loaded tensors
    - Losses live on different GPUs (accelerate sharded the 30B model) — moved
      to common device before composition
    - Trainable param count is ~710M; checkpoints saved as state dict only

Public API:
    run_training(model, dataloader, *, num_steps, ...)

Loss composition:
    loss = nll
         + beta_v * kl_v + beta_a * kl_a + beta_j * kl_j
         + aux_weight * (nll_aux_v + nll_aux_a)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn


def trainable_state_dict(model: nn.Module) -> dict:
    """Return only parameters with requires_grad=True. ~710M for v5 vs 30B full."""
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _compose_loss(nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j,
                  *, beta_v: float, beta_a: float, beta_j: float, aux_weight: float):
    """Combine 6 losses on potentially-different devices into one scalar on nll's device."""
    dev = nll.device
    return (nll
            + beta_v * kl_v.to(dev)
            + beta_a * kl_a.to(dev)
            + beta_j * kl_j.to(dev)
            + aux_weight * (nll_aux_v.to(dev) + nll_aux_a.to(dev)))


def run_training(
    model: nn.Module,
    dataloader: Iterable,
    *,
    num_steps: int,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    grad_clip: float = 1.0,
    beta_v: float = 0.0,
    beta_a: float = 0.0,
    beta_j: float = 0.0,
    aux_weight: float = 0.1,
    log_path: str | Path = "train_log.jsonl",
    ckpt_path: Optional[str | Path] = None,   # if set, save final ckpt here
    print_every: int = 1,
) -> dict:
    """Train v5 for num_steps. Returns summary dict."""
    log_path = Path(log_path)
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        trainable_params(model),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    model.train()
    log_f = open(log_path, "w")

    def cycle(loader):
        while True:
            for b in loader:
                yield b

    it = cycle(dataloader)
    step = 0
    t0 = time.time()
    print(f"Training {num_steps} steps. betas=(v={beta_v}, a={beta_a}, j={beta_j}), aux_w={aux_weight}, lr={lr}")

    while step < num_steps:
        batch = next(it)
        # Batch shape: each field is a list of length B (B=1 in our case)
        videos = batch["videos"]
        audios = batch["audios"]
        prompts = batch["prompts"]
        answers = batch["answers"]

        nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j = model.forward_train(
            videos, audios, prompts, answers,
        )
        loss = _compose_loss(
            nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j,
            beta_v=beta_v, beta_a=beta_a, beta_j=beta_j, aux_weight=aux_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params(model), grad_clip)
        optimizer.step()

        rec = {
            "step": step,
            "loss": float(loss.item()),
            "nll": float(nll.item()),
            "nll_aux_v": float(nll_aux_v.item()),
            "nll_aux_a": float(nll_aux_a.item()),
            "kl_v": float(kl_v.item()),
            "kl_a": float(kl_a.item()),
            "kl_j": float(kl_j.item()),
            "grad_norm": float(grad_norm),
            "lr": lr,
            "elapsed_s": time.time() - t0,
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

        if step % print_every == 0:
            print(f"  step {step:4d}  loss={rec['loss']:7.3f}  nll={rec['nll']:6.3f}  "
                  f"kl=({rec['kl_v']:.0f},{rec['kl_a']:.0f},{rec['kl_j']:.0f})  "
                  f"gn={rec['grad_norm']:.2f}  t={rec['elapsed_s']:.0f}s",
                  flush=True)

        step += 1

    log_f.close()
    elapsed = time.time() - t0
    print(f"\nTraining complete: {num_steps} steps in {elapsed:.1f}s ({num_steps/elapsed:.2f} steps/s)")

    if ckpt_path is not None:
        print(f"Saving final trainable state to {ckpt_path}")
        torch.save(
            {"step": num_steps - 1, "trainable_state": trainable_state_dict(model)},
            ckpt_path,
        )
        print("  saved.")

    return {"num_steps": num_steps, "elapsed_s": elapsed, "final_loss": rec["loss"]}
