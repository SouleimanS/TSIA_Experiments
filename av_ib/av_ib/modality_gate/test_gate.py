"""Lightweight checks for the modality gate math (no 30B backbone needed).

    python -m av_ib.modality_gate.test_gate
"""
from __future__ import annotations

import torch

from av_ib.modality_gate.gate import ModalityGate, MODALITY_NAMES


def test_zero_init_is_uniform():
    """Zero-init final layer -> uniform distribution for any input."""
    gate = ModalityGate(d_model=2048, hidden=128)
    x = torch.randn(4, 2048)
    p = gate(x)
    assert p.shape == (4, len(MODALITY_NAMES))
    torch.testing.assert_close(
        p, torch.full_like(p, 1.0 / len(MODALITY_NAMES)), atol=1e-6, rtol=0
    )
    print("ok: zero-init gate is uniform ->", p[0].tolist())


def test_centered_uniform_is_identity():
    """At uniform p, the centred scale (M*p) equals 1.0 for every modality."""
    M = len(MODALITY_NAMES)
    p = torch.full((1, M), 1.0 / M)
    scale = M * p
    torch.testing.assert_close(scale, torch.ones_like(scale))
    print("ok: M*p at uniform == 1.0 (identity scaling)")


def test_softmax_is_distribution():
    gate = ModalityGate(d_model=2048, hidden=128)
    # Perturb weights so output is non-uniform
    with torch.no_grad():
        gate.net[-1].weight.normal_(0, 1.0)
        gate.net[-1].bias.normal_(0, 1.0)
    p = gate(torch.randn(8, 2048))
    sums = p.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=0)
    assert (p >= 0).all()
    print("ok: gate output is a valid distribution (sums to 1, non-negative)")


def test_gradient_flows_to_gate():
    """A loss on the scaled embedding must backprop into the gate."""
    gate = ModalityGate(d_model=2048, hidden=128)
    with torch.no_grad():
        gate.net[-1].weight.normal_(0, 0.1)
    rep = torch.randn(1, 2048)
    p = gate(rep)
    M = p.shape[-1]
    z = torch.randn(1, 10, 2048)            # fake injected tokens
    z_scaled = z * (M * p[0, 0])            # scale by centred video prob
    loss = z_scaled.pow(2).mean()
    loss.backward()
    g = gate.net[-1].weight.grad
    assert g is not None and g.abs().sum() > 0
    print("ok: gradient reaches gate MLP (sum|grad|=%.4f)" % g.abs().sum())


if __name__ == "__main__":
    test_zero_init_is_uniform()
    test_centered_uniform_is_identity()
    test_softmax_is_distribution()
    test_gradient_flows_to_gate()
    print("\nAll gate math checks passed.")
