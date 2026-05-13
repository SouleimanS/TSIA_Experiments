"""Smoke test for av_ib.model.qformer."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from av_ib.model.qformer import VideoQFormer, AudioQFormer, count_trainable


def test_video_qformer():
    print("=== Building VideoQFormer ===")
    qf = VideoQFormer().cuda()
    trainable, total = count_trainable(qf)
    print(f"  params: total={total/1e6:.1f}M trainable={trainable/1e6:.1f}M")
    eva = torch.randn(8, 257, 1408, device="cuda")
    print(f"  input shape: {tuple(eva.shape)}")
    out = qf(eva)
    print(f"  output shape: {tuple(out.shape)} dtype={out.dtype}")
    expected = (1, 32, 4096)
    assert tuple(out.shape) == expected, f"Expected {expected}, got {tuple(out.shape)}"
    print("  OK\n")


def test_audio_qformer():
    print("=== Building AudioQFormer ===")
    qf = AudioQFormer().cuda()
    trainable, total = count_trainable(qf)
    print(f"  params: total={total/1e6:.1f}M trainable={trainable/1e6:.1f}M")
    feats = torch.randn(1, 8, 1024, device="cuda")
    print(f"  input shape: {tuple(feats.shape)}")
    out = qf(feats)
    print(f"  output shape: {tuple(out.shape)} dtype={out.dtype}")
    expected = (1, 8, 4096)
    assert tuple(out.shape) == expected, f"Expected {expected}, got {tuple(out.shape)}"
    print("  OK\n")


if __name__ == "__main__":
    test_video_qformer()
    test_audio_qformer()
    print("All Q-Former smoke tests passed.")
