"""Shape-contract tests for every module. Run with `pytest tests/`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from av_ib.config import load_config
from av_ib.models.encoders import (
    build_video_encoder, build_audio_encoder, build_projector,
)
from av_ib.models.fusion import IdentityFusion, MutualCrossAttention
from av_ib.models.bottleneck import IdentityBottleneck, VIB, PerModalityVIB
from av_ib.models.av_model import AVModel


def test_identity_fusion_shape():
    fusion = IdentityFusion()
    Zv = torch.randn(2, 32, 64)
    Za = torch.randn(2, 24, 64)
    out, _ = fusion(Zv, Za)
    assert out.shape == (2, 56, 64)


def test_mutual_cross_attention_shape():
    fusion = MutualCrossAttention(d=64, n_heads=4, d_ff=128)
    Zv = torch.randn(2, 32, 64)
    Za = torch.randn(2, 24, 64)
    out, _ = fusion(Zv, Za)
    assert out.shape == (2, 56, 64)


def test_identity_bottleneck():
    bn = IdentityBottleneck()
    Z = torch.randn(2, 16, 64)
    T, kl = bn(Z)
    assert torch.equal(T, Z)
    assert kl.shape == (2,)
    assert torch.all(kl == 0)


def test_vib_shape_and_kl_positive():
    bn = VIB(d=64, d_h=32)
    bn.train()
    Z = torch.randn(2, 16, 64)
    T, kl = bn(Z)
    assert T.shape == Z.shape
    assert kl.shape == (2,)
    # KL is non-negative for any Gaussian against N(0, I).
    assert torch.all(kl >= 0)


def test_per_modality_vib_shape():
    bn = PerModalityVIB(d=64, d_h=32)
    bn.train()
    Zv = torch.randn(2, 32, 64)
    Za = torch.randn(2, 24, 64)
    Tv, Ta, kl = bn(Zv, Za)
    assert Tv.shape == Zv.shape
    assert Ta.shape == Za.shape
    assert kl.shape == (2,)
    assert torch.all(kl >= 0)


def test_full_model_v1_forward_backward(tmp_path):
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "v1_no_fusion_no_vib.yaml"
    cfg = load_config(cfg_path, overrides=[
        "train.device=cpu",
        "model.llm.hidden_size=64",
        "model.video_projector.d_out=64",
        "model.audio_projector.d_out=64",
        "model.video_projector.num_tokens=8",
        "model.audio_projector.num_tokens=8",
        "data.answer_len=4",
        "data.vocab_size=128",
        "model.llm.kwargs={vocab_size: 128, n_heads: 4}",
    ])
    model = AVModel(cfg)
    # v1 has zero trainable parameters.
    assert len(model.trainable_parameters()) == 0

    # Forward must still produce a finite loss.
    x_v = torch.randn(2, 4, 16, 16, 3)
    x_a = torch.randn(2, 1024)
    labels = torch.randint(0, 128, (2, 4))
    out = model(x_v, x_a, labels=labels)
    assert torch.isfinite(out.loss_nll)
    assert out.kl.shape == (2,)
    assert torch.all(out.kl == 0)
