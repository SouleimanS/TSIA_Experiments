"""Smoke test for av_ib.model.encoders.

What it checks:
  1. Imports resolve (sys.path trick works).
  2. EVA-ViT loads from disk without error.
  3. ImageBind loads from disk without error.
  4. Both produce outputs of the expected shape on random input.
  5. Both report zero trainable parameters (frozen).

If any of these fail, we fix it before writing more code.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the av_ib package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from av_ib.model.encoders import VideoEncoder, AudioEncoder, freeze_count


def test_video():
    print("=== Loading VideoEncoder (EVA-ViT-G) ===")
    enc = VideoEncoder(precision="fp16").cuda()
    trainable, total = freeze_count(enc)
    print(f"  params: total={total/1e6:.1f}M trainable={trainable}")
    assert trainable == 0, "Video encoder must be fully frozen"

    # 1 video, 8 frames, 3 channels, 224x224. fp16 to match the encoder dtype.
    x = torch.randn(1, 8, 3, 224, 224, dtype=torch.float16, device="cuda")
    print(f"  input shape: {tuple(x.shape)}")
    y = enc(x)
    print(f"  output shape: {tuple(y.shape)} dtype={y.dtype}")
    expected = (8, 257, 1408)   # B*T=1*8=8, N_tok=257, d=1408
    assert tuple(y.shape) == expected, f"Expected {expected}, got {tuple(y.shape)}"
    print("  OK\n")


def test_audio():
    print("=== Loading AudioEncoder (ImageBind) ===")
    enc = AudioEncoder().cuda()
    trainable, total = freeze_count(enc)
    print(f"  params: total={total/1e6:.1f}M trainable={trainable}")
    assert trainable == 0, "Audio encoder must be fully frozen"

    # Pre-loaded mel: (B=1, N_clips=3, C=1, mel_bins=128, time=204).
    # 3 clips matches the default ImageBind sampler (clips_per_video=3).
    mel = torch.randn(1, 8, 1, 128, 204, device="cuda")
    print(f"  input mel shape: {tuple(mel.shape)}")
    y = enc.forward_from_mel(mel)
    print(f"  output shape: {tuple(y.shape)} dtype={y.dtype}")
    expected = (1, 8, 1024)
    assert tuple(y.shape) == expected, f"Expected {expected}, got {tuple(y.shape)}"
    print("  OK\n")


if __name__ == "__main__":
    test_video()
    test_audio()
    print("All encoder smoke tests passed.")
