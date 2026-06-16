"""Train one Video-IB autoencoder at a fixed beta.

Loss per step:  L = D + beta * R
    D = reconstruction_distortion(x, x_hat)       (distortion)
    R = kl_per_dim.sum(dim=1).mean()              (rate, nats per clip)

Logs rate and distortion separately so a sweep over beta can plot the
rate-distortion (IB) curve. Usable as a library call (`train(cfg)`) or a CLI.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import MovingShapes
from .model import (
    ModelConfig,
    VideoIBAutoencoder,
    reconstruction_distortion,
    psnr,
)


@dataclass
class TrainConfig:
    beta: float = 1e-3
    latent_dim: int = 16
    width: int = 32
    # data
    n_train: int = 1024
    n_eval: int = 256
    T: int = 8
    H: int = 32
    W: int = 32
    C: int = 1
    # optimisation
    steps: int = 1500
    batch_size: int = 32
    lr: float = 2e-3
    # bookkeeping
    seed: int = 0
    device: str = "cpu"
    out_dir: str | None = None
    log_every: int = 50
    active_unit_threshold: float = 1e-2  # nats; KL/dim above this = "active"


def _active_units(model, loader, device, threshold) -> int:
    """How many latent dimensions carry real information (mean KL > threshold)."""
    model.eval()
    tot = None
    n = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            _, kl_per_dim = model(x)
            s = kl_per_dim.sum(dim=0)
            tot = s if tot is None else tot + s
            n += x.shape[0]
    mean_kl = tot / max(n, 1)
    return int((mean_kl > threshold).sum().item())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    d_sum, r_sum, p_sum, n = 0.0, 0.0, 0.0, 0
    for x in loader:
        x = x.to(device)
        x_hat, kl_per_dim = model(x)
        b = x.shape[0]
        d_sum += reconstruction_distortion(x, x_hat).item() * b
        r_sum += kl_per_dim.sum(dim=1).mean().item() * b
        p_sum += psnr(x, x_hat) * b
        n += b
    return {"distortion": d_sum / n, "rate": r_sum / n, "psnr": p_sum / n}


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_ds = MovingShapes(cfg.n_train, cfg.T, cfg.H, cfg.W, cfg.C, seed=cfg.seed)
    eval_ds = MovingShapes(cfg.n_eval, cfg.T, cfg.H, cfg.W, cfg.C, seed=cfg.seed + 999)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size)

    mcfg = ModelConfig(C=cfg.C, T=cfg.T, H=cfg.H, W=cfg.W,
                       latent_dim=cfg.latent_dim, width=cfg.width)
    model = VideoIBAutoencoder(mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    step = 0
    t0 = time.time()
    data_iter = iter(train_loader)
    while step < cfg.steps:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x = next(data_iter)
        x = x.to(device)

        model.train()
        x_hat, kl_per_dim = model(x)
        distortion = reconstruction_distortion(x, x_hat)
        rate = kl_per_dim.sum(dim=1).mean()
        loss = distortion + cfg.beta * rate

        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1

        if step % cfg.log_every == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), "distortion": distortion.item(),
                   "rate": rate.item(), "elapsed": round(time.time() - t0, 1)}
            history.append(rec)
            print(f"[beta={cfg.beta:g}] step {step:5d}  loss {rec['loss']:9.2f}  "
                  f"D {rec['distortion']:9.2f}  R {rec['rate']:7.3f}  "
                  f"({rec['elapsed']}s)")

    final = evaluate(model, eval_loader, device)
    final["active_units"] = _active_units(model, eval_loader, device,
                                          cfg.active_unit_threshold)
    final["beta"] = cfg.beta
    print(f"[beta={cfg.beta:g}] FINAL  rate {final['rate']:.3f} nats  "
          f"PSNR {final['psnr']:.2f} dB  active_units "
          f"{final['active_units']}/{cfg.latent_dim}")

    result = {"config": asdict(cfg), "final": final, "history": history}
    if cfg.out_dir:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        torch.save(model.state_dict(), out / "model.pt")
    result["_model"] = model
    result["_eval_loader"] = eval_loader
    return result


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a Video-IB autoencoder.")
    for field, default in asdict(TrainConfig()).items():
        if default is None:
            p.add_argument(f"--{field}", default=None)
        elif isinstance(default, bool):
            p.add_argument(f"--{field}", type=lambda s: s.lower() == "true",
                           default=default)
        else:
            p.add_argument(f"--{field}", type=type(default), default=default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
