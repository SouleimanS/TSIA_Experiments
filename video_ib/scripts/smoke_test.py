"""Fast end-to-end check that the encode -> IB -> decode pipeline runs and learns.

Tiny config, CPU-friendly (~1 minute). Verifies:
  1. shapes flow through encoder -> VIB -> decoder,
  2. the loss goes down,
  3. a high-beta run keeps fewer active latent units than a low-beta run
     (the bottleneck actually bottlenecks).

Run:  python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from video_ib.data import make_moving_shapes
from video_ib.model import ModelConfig, VideoIBAutoencoder, reconstruction_distortion
from video_ib.train import TrainConfig, train


def test_forward_shapes():
    cfg = ModelConfig(C=1, T=8, H=32, W=32, latent_dim=16, width=16)
    model = VideoIBAutoencoder(cfg)
    x = make_moving_shapes(4, T=8, H=32, W=32, C=1, seed=1)
    x_hat, kl = model(x)
    assert x_hat.shape == x.shape, (x_hat.shape, x.shape)
    assert kl.shape == (4, 16), kl.shape
    assert torch.isfinite(x_hat).all() and torch.isfinite(kl).all()
    print(f"[ok] forward shapes: x_hat {tuple(x_hat.shape)}, kl {tuple(kl.shape)}")


def test_learns_and_bottlenecks():
    common = dict(latent_dim=16, width=16, n_train=128, n_eval=64, T=8, H=32, W=32,
                  steps=200, batch_size=16, lr=2e-3, log_every=50)
    lo = train(TrainConfig(beta=1e-4, **common))
    hi = train(TrainConfig(beta=1e-1, **common))

    first = lo["history"][0]["loss"]
    last = lo["history"][-1]["loss"]
    assert last < first, f"loss did not decrease: {first:.1f} -> {last:.1f}"
    print(f"[ok] loss decreased {first:.1f} -> {last:.1f}")

    # Tighter bottleneck should keep no more information than a loose one.
    assert hi["final"]["rate"] <= lo["final"]["rate"] + 1e-6, (
        hi["final"]["rate"], lo["final"]["rate"])
    assert hi["final"]["active_units"] <= lo["final"]["active_units"], (
        hi["final"]["active_units"], lo["final"]["active_units"])
    print(f"[ok] IB compresses: rate {lo['final']['rate']:.2f}->"
          f"{hi['final']['rate']:.2f} nats, "
          f"active units {lo['final']['active_units']}->"
          f"{hi['final']['active_units']}, "
          f"PSNR {lo['final']['psnr']:.1f}->{hi['final']['psnr']:.1f} dB")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_forward_shapes()
    test_learns_and_bottlenecks()
    print("\nALL SMOKE CHECKS PASSED")
