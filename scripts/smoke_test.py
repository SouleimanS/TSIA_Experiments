"""Smoke test entry point.

Usage:
    python scripts/smoke_test.py --config configs/v1_no_fusion_no_vib.yaml
    python scripts/smoke_test.py --config configs/v1_no_fusion_no_vib.yaml train.device=cpu

What it does:
    1. Load the config.
    2. Build the AV model.
    3. Print the trainable-module breakdown so you can verify which
       modules are receiving gradient.
    4. Run `train.num_steps` training steps on dummy data.
    5. Print loss, NLL, KL, grad-norm per step.

For variant v1 (no fusion, no VIB), the trainable-module breakdown
should show 0 trainable parameters in every module. The forward pass
still runs and the LLM loss is computed; backward is a no-op.

For variants that introduce trainable modules (cross-attention, VIB),
the breakdown should show non-zero counts only for those modules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/smoke_test.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from av_ib.config import load_config, runcfg_to_dict
from av_ib.data.dummy import build_dummy_loader
from av_ib.models.av_model import AVModel
from av_ib.training.train_step import train_step


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument(
        "overrides", nargs="*",
        help="dotted.key=value overrides applied after YAML load.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    print("=" * 72)
    print("CONFIG")
    print("=" * 72)
    print(json.dumps(runcfg_to_dict(cfg), indent=2))
    print()

    # Device resolution. If CUDA is requested but unavailable, fall back to CPU
    # with a loud warning. This makes the smoke test runnable on a laptop too.
    device_str = cfg.train.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: device={device_str} requested but CUDA is unavailable; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    torch.manual_seed(cfg.train.seed)

    print("=" * 72)
    print("BUILD MODEL")
    print("=" * 72)
    model = AVModel(cfg).to(device)
    summary = model.trainable_module_summary()
    total_trainable = sum(summary.values())
    print(f"Total trainable parameters: {total_trainable:,}")
    for name, n in summary.items():
        print(f"  {name:30s} {n:>12,}")
    print()

    # Build optimizer over trainable params only. If there are none (variant v1),
    # we still want the loop to run, so we attach a dummy parameter that holds
    # no gradient (placeholder so torch.optim doesn't complain about empty params).
    trainable = model.trainable_parameters()
    if len(trainable) == 0:
        print("No trainable parameters -- optimizer will be a no-op.")
        # Make a throwaway parameter just so the optimizer can be constructed.
        # It is detached from the graph; backward does nothing to it.
        dummy = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
        optimizer = torch.optim.AdamW([dummy], lr=cfg.train.lr)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr)

    print("=" * 72)
    print("RUN STEPS")
    print("=" * 72)
    loader = build_dummy_loader(cfg.data, batch_size=cfg.train.batch_size, n_examples=cfg.train.batch_size * cfg.train.num_steps)
    iter_loader = iter(loader)

    for step in range(cfg.train.num_steps):
        try:
            batch = next(iter_loader)
        except StopIteration:
            # Reset if num_steps > number of examples in dummy dataset.
            iter_loader = iter(loader)
            batch = next(iter_loader)

        stats = train_step(model, batch, optimizer, beta=cfg.train.beta, device=device)
        print(f"step {step:04d}  " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()))

    print()
    print("=" * 72)
    print("SMOKE TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
