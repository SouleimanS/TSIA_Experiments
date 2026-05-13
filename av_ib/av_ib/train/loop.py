"""Single-GPU training loop for AVModelV1 (and future variants).

Public API:
    run_training(model, dataloader, *, num_steps, lr, log_path, ckpt_dir, ...)

What it does:
    - Builds an AdamW optimizer over model.trainable_parameters_grouped()
      (we expose this on the model so we can use different LRs for
       Q-Formers vs LoRA later, but for now one group at one LR).
    - For each batch: forward_train -> backward -> clip -> step.
    - Logs to stdout AND a JSONL file (one record per step).
    - Saves a checkpoint of the trainable state every save_every steps
      (full model state would be huge, but we only need the 310M trainable
       params plus the optimizer state).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torch import nn


def trainable_state_dict(model: nn.Module) -> dict:
    """Return only parameters with requires_grad=True. Saves disk space:
    ~1.2GB instead of ~36GB for the full model."""
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def run_training(
    model: nn.Module,
    dataloader: Iterable,
    *,
    num_steps: int,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    grad_clip: float = 1.0,
    log_path: str | Path = "train_log.jsonl",
    ckpt_dir: str | Path = "ckpts",
    save_every: int = 0,                # 0 = don't save
    device: str = "cuda",
    print_every: int = 1,
) -> None:
    """Train `model` for `num_steps` steps on `dataloader`.

    The dataloader is iterated repeatedly if num_steps > len(dataloader).
    """
    log_path = Path(log_path)
    ckpt_dir = Path(ckpt_dir)
    if save_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # AdamW on trainable params only.
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

    while step < num_steps:
        batch = next(it)
        videos = batch["videos"].to(device, non_blocking=True)
        audio_mels = batch["audio_mels"].to(device, non_blocking=True)
        prompts = batch["prompts"]
        answers = batch["answers"]

        loss = model.forward_train(videos, audio_mels, prompts, answers)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params(model), grad_clip)
        optimizer.step()

        rec = {
            "step": step,
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "lr": lr,
            "elapsed_s": time.time() - t0,
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if step % print_every == 0:
            print(f"  step {step:4d}  loss={rec['loss']:.4f}  "
                  f"grad_norm={rec['grad_norm']:.2f}  "
                  f"elapsed={rec['elapsed_s']:.1f}s")

        if save_every > 0 and (step + 1) % save_every == 0:
            ckpt = {
                "step": step,
                "trainable_state": trainable_state_dict(model),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(ckpt, ckpt_dir / f"step_{step:06d}.pt")
            print(f"  saved {ckpt_dir / f'step_{step:06d}.pt'}")

        step += 1

    log_f.close()
    print(f"\nTraining complete: {num_steps} steps in {time.time() - t0:.1f}s")
