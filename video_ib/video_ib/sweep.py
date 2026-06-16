"""Sweep the bottleneck strength beta -- this is the experiment.

For each beta we train a fresh autoencoder and record:
    - rate  R  (nats/clip): how much information the latent keeps
    - distortion / PSNR    : how well the clip is reconstructed
    - active units         : how many latent dimensions stay in use

Outputs (under --out_dir):
    sweep.json / sweep.csv          : the numbers
    rate_distortion.png             : the IB curve (rate vs PSNR), if matplotlib
    reconstructions.png             : a clip decoded at each beta, if matplotlib

The rate-distortion curve and the reconstruction strip together *are* "the
effect of the IB": as beta rises, rate and active units fall, and you watch the
decode shed detail -- noise and fine texture first, coarse structure last.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch

from .train import TrainConfig, train

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - plotting is optional
    _HAVE_MPL = False


def _plot_rate_distortion(rows, out_path):
    rates = [r["rate"] for r in rows]
    psnrs = [r["psnr"] for r in rows]
    betas = [r["beta"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(rates, psnrs, "-o", color="#2b6cb0")
    for x, y, b in zip(rates, psnrs, betas):
        ax.annotate(f"β={b:g}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("rate  R = KL(q(z|x) || p(z))   [nats/clip]  →  more info kept")
    ax.set_ylabel("PSNR [dB]  →  better reconstruction")
    ax.set_title("Information Bottleneck: rate–distortion curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_reconstructions(models_and_betas, eval_clip, out_path, frame_idx=None):
    """Decode one held-out clip at every beta and show a few frames each."""
    C, T, H, W = eval_clip.shape
    if frame_idx is None:
        frame_idx = [0, T // 2, T - 1]
    n_rows = len(models_and_betas) + 1  # +1 for the ground truth row
    n_cols = len(frame_idx)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))
    if n_cols == 1:
        axes = axes[:, None]

    def show(ax, frame, title=None):
        ax.imshow(frame.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_ylabel(title, fontsize=9)

    for j, fi in enumerate(frame_idx):
        show(axes[0][j], eval_clip[0, fi])
        if j == 0:
            axes[0][j].set_ylabel("ground truth", fontsize=9)
        axes[0][j].set_title(f"t={fi}", fontsize=9)

    x = eval_clip.unsqueeze(0)
    for i, (model, beta) in enumerate(models_and_betas, start=1):
        model.eval()
        with torch.no_grad():
            x_hat, _ = model(x)
        for j, fi in enumerate(frame_idx):
            show(axes[i][j], x_hat[0, 0, fi])
            if j == 0:
                axes[i][j].set_ylabel(f"β={beta:g}", fontsize=9)
    fig.suptitle("Same clip decoded through tighter bottlenecks (top→bottom)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_sweep(betas, base: TrainConfig, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    models_and_betas = []
    eval_clip = None
    for beta in betas:
        cfg = replace(base, beta=beta, out_dir=str(out / f"beta_{beta:g}"))
        res = train(cfg)
        rows.append(res["final"])
        models_and_betas.append((res["_model"], beta))
        if eval_clip is None:
            eval_clip = next(iter(res["_eval_loader"]))[0]

    rows.sort(key=lambda r: r["beta"])
    with open(out / "sweep.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["beta", "rate", "distortion", "psnr",
                                          "active_units"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    print("\n=== sweep summary ===")
    print(f"{'beta':>10} {'rate(nats)':>12} {'PSNR(dB)':>10} {'active':>8}")
    for r in rows:
        print(f"{r['beta']:>10g} {r['rate']:>12.3f} {r['psnr']:>10.2f} "
              f"{r['active_units']:>8d}")

    if _HAVE_MPL:
        _plot_rate_distortion(rows, out / "rate_distortion.png")
        _plot_reconstructions(models_and_betas, eval_clip,
                              out / "reconstructions.png")
        print(f"\nwrote plots to {out}/rate_distortion.png and "
              f"{out}/reconstructions.png")
    else:
        print("\n(matplotlib not installed -> skipped PNGs; numbers are in "
              "sweep.json / sweep.csv)")
    return rows


def _parse_args():
    p = argparse.ArgumentParser(description="Sweep beta for the Video-IB study.")
    p.add_argument("--betas", type=str,
                   default="0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2",
                   help="comma-separated list of beta values")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_dir", type=str, default="outputs/sweep")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    betas = [float(b) for b in a.betas.split(",")]
    base = TrainConfig(steps=a.steps, latent_dim=a.latent_dim, T=a.T, H=a.H,
                       W=a.W, n_train=a.n_train, device=a.device)
    run_sweep(betas, base, a.out_dir)
