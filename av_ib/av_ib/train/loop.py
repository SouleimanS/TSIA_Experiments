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
                  *, beta_v: float, beta_a: float, beta_j: float, aux_weight: float,
                  model_handles_betas: bool = False):
    """Combine 6 losses into a scalar on nll's device.

    If model_handles_betas=True (v6 sink-aware), the returned kl_v/kl_a/kl_j
    are already weighted by betas inside the model; multiply by 1.0 here.
    Otherwise (v5 and earlier), multiply by the given beta_v/beta_a/beta_j.
    """
    dev = nll.device
    mul_v = 1.0 if model_handles_betas else beta_v
    mul_a = 1.0 if model_handles_betas else beta_a
    mul_j = 1.0 if model_handles_betas else beta_j
    return (nll
            + mul_v * kl_v.to(dev)
            + mul_a * kl_a.to(dev)
            + mul_j * kl_j.to(dev)
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
    save_every: int = 0,   # if >0, save step_N.pt every N steps
    model_handles_betas: bool = False,
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
            model_handles_betas=model_handles_betas,
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
        # Sink/noise diagnostics (only present for AVModelV6, not the baseline)
        diag = getattr(model, "last_diagnostics", None)
        if isinstance(diag, dict):
            rec.update(diag)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

        if step % print_every == 0:
            extra = ""
            if isinstance(diag, dict) and "sink_frac_v" in diag:
                extra = (f"  sink_v={diag['sink_frac_v']:.2f}"
                         f"  std_v={diag.get('std_nonsink_v', float('nan')):.3f}"
                         f"  std_a={diag.get('std_nonsink_a', float('nan')):.3f}")
            print(f"  step {step:4d}  loss={rec['loss']:7.3f}  nll={rec['nll']:6.3f}  "
                  f"kl=({rec['kl_v']:.0f},{rec['kl_a']:.0f},{rec['kl_j']:.0f})  "
                  f"gn={rec['grad_norm']:.2f}{extra}  t={rec['elapsed_s']:.0f}s",
                  flush=True)

        # Periodic checkpoint (every save_every steps, if save_every > 0)
        if save_every > 0 and ckpt_path is not None and (step + 1) % save_every == 0:
            periodic_path = ckpt_path.parent / f"step_{step+1}.pt"
            torch.save(
                {"step": step, "trainable_state": trainable_state_dict(model)},
                periodic_path,
            )
            print(f"  [ckpt] saved {periodic_path.name}", flush=True)

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
