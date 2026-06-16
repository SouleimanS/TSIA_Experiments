"""Synthetic video data with a small, known set of generative factors.

Why synthetic? To study *the effect of the Information Bottleneck* we want clips
whose information content we understand exactly. Each clip here is fully
described by a handful of latent factors:

    - shape type (square or disk)
    - radius / size
    - brightness
    - initial position (x, y)
    - velocity (vx, vy)   -- the shape moves at constant speed and bounces
                             off the frame edges

A perfect encoder needs only ~7 numbers to describe a whole clip. So as we
tighten the bottleneck (increase beta) we can literally watch the latent code
shed factors -- fine detail (exact brightness, size) goes first, coarse
structure (where the blob is, that there *is* a blob) survives longest. That is
the IB effect made visible.

No downloads, no external assets: clips are rasterised with NumPy on the fly.
A real-video drop-in (e.g. Moving MNIST, UCF101, Kinetics) only needs to return
the same (T, C, H, W) float tensor in [0, 1]; nothing else in the pipeline cares
where the pixels came from.

Tensor layout returned by the dataset: (C, T, H, W), float32 in [0, 1].
Batched by the default collate this becomes (B, C, T, H, W), which is exactly
what torch.nn.Conv3d expects (channels first, then the temporal "depth" axis).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def _render_clip(
    rng: np.random.Generator,
    T: int,
    H: int,
    W: int,
    C: int,
) -> np.ndarray:
    """Render a single clip of a moving shape. Returns (C, T, H, W) in [0, 1]."""
    shape_is_disk = rng.random() < 0.5
    radius = rng.uniform(0.10, 0.22) * min(H, W)
    brightness = rng.uniform(0.6, 1.0)

    # Position / velocity in pixel units; keep the shape fully on-screen.
    margin = radius + 1.0
    cx = rng.uniform(margin, W - margin)
    cy = rng.uniform(margin, H - margin)
    speed = rng.uniform(0.05, 0.18) * min(H, W)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    vx, vy = speed * np.cos(angle), speed * np.sin(angle)

    ys = np.arange(H, dtype=np.float32)[:, None]   # (H, 1)
    xs = np.arange(W, dtype=np.float32)[None, :]   # (1, W)

    frames = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        # Bounce off the walls so the trajectory stays on-screen and non-trivial.
        if cx < margin or cx > W - margin:
            vx = -vx
            cx = float(np.clip(cx, margin, W - margin))
        if cy < margin or cy > H - margin:
            vy = -vy
            cy = float(np.clip(cy, margin, H - margin))

        if shape_is_disk:
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        else:
            mask = (np.abs(xs - cx) <= radius) & (np.abs(ys - cy) <= radius)
        frames[t][mask] = brightness

        cx += vx
        cy += vy

    clip = np.repeat(frames[None, :, :, :], C, axis=0)  # (C, T, H, W)
    return clip


def make_moving_shapes(
    n: int,
    T: int = 8,
    H: int = 32,
    W: int = 32,
    C: int = 1,
    seed: int = 0,
) -> torch.Tensor:
    """Generate ``n`` clips. Returns a float tensor of shape (n, C, T, H, W)."""
    rng = np.random.default_rng(seed)
    clips = np.stack([_render_clip(rng, T, H, W, C) for _ in range(n)], axis=0)
    return torch.from_numpy(clips).float()


class MovingShapes(Dataset):
    """In-memory dataset of synthetic moving-shape clips.

    Pre-rendered once at construction so repeated epochs see identical data
    (deterministic given ``seed``), which keeps rate/distortion numbers
    comparable across beta values in a sweep.
    """

    def __init__(
        self,
        n: int = 1024,
        T: int = 8,
        H: int = 32,
        W: int = 32,
        C: int = 1,
        seed: int = 0,
    ) -> None:
        self.clips = make_moving_shapes(n, T=T, H=H, W=W, C=C, seed=seed)
        self.T, self.H, self.W, self.C = T, H, W, C

    def __len__(self) -> int:
        return self.clips.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.clips[idx]  # (C, T, H, W)
