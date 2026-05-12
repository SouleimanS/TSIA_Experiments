"""Fusion modules. Each takes (Zv, Za) and returns Zfused.

Variants:
    IdentityFusion -- just concatenates along the token axis. This is the
        'no cross-attention' baseline: the LLM's own self-attention is the
        only mechanism that mixes modalities. Used for variant 1.

    MutualCrossAttention -- the directive's Φ (eqs. 3-4 in the whitepaper).
        Two parallel cross-attention blocks (v->a and a->v) with residual
        connections and per-modality FFNs, then token-axis concat.

The contract is the same for both:

    forward(Zv, Za) -> (Zfused, info_dict)

where:
    Zv:     (B, Nv, d)
    Za:     (B, Na, d)
    Zfused: (B, Nv + Na, d)         -- concatenation along token axis
    info_dict: optional diagnostics (e.g. attention weights) for analysis.

Token-axis concat (not feature-axis) is chosen for the reasons in section
3.3 of the whitepaper: preserves LLM input contract, no implicit token
alignment between modalities, lets LLM self-attention do further fusion.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor


class IdentityFusion(nn.Module):
    """No learned fusion. Concatenates Zv and Za along the token axis."""

    def __init__(self):
        super().__init__()

    def forward(self, Zv: Tensor, Za: Tensor) -> tuple[Tensor, dict]:
        assert Zv.dim() == 3 and Za.dim() == 3, "Z* must be (B, N, d)"
        assert Zv.size(-1) == Za.size(-1), (
            f"Hidden dims must match for concat: {Zv.size(-1)} vs {Za.size(-1)}"
        )
        Zfused = torch.cat([Zv, Za], dim=1)
        return Zfused, {}


class _CrossAttnBlock(nn.Module):
    """One direction of Φ: queries from one modality, keys/values from the
    other. Residual + LN + FFN. The output keeps the query modality's
    token count (so the v->a block emits Nv tokens, a->v emits Na)."""

    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d),
        )
        self.ln2 = nn.LayerNorm(d)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        attn_out, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        h = self.ln1(q + attn_out)
        h = self.ln2(h + self.ffn(h))
        return h


class MutualCrossAttention(nn.Module):
    """Φ: mutual cross-attention between Zv and Za.

    Implements eqs. 3-4 of the whitepaper. Each modality attends to the
    other; both outputs are then concatenated along the token axis.
    """

    def __init__(self, d: int, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.v_to_a = _CrossAttnBlock(d, n_heads, d_ff, dropout)   # Q=Zv, KV=Za
        self.a_to_v = _CrossAttnBlock(d, n_heads, d_ff, dropout)   # Q=Za, KV=Zv

    def forward(self, Zv: Tensor, Za: Tensor) -> tuple[Tensor, dict]:
        Zv_enriched = self.v_to_a(q=Zv, kv=Za)    # (B, Nv, d)
        Za_enriched = self.a_to_v(q=Za, kv=Zv)    # (B, Na, d)
        Zfused = torch.cat([Zv_enriched, Za_enriched], dim=1)
        return Zfused, {}


_FUSION_REGISTRY = {
    "identity": lambda d, **kw: IdentityFusion(),
    "mutual_cross_attention": lambda d, **kw: MutualCrossAttention(d=d, **kw),
}


def build_fusion(name: str, d: int, **kwargs) -> nn.Module:
    if name not in _FUSION_REGISTRY:
        raise KeyError(f"Unknown fusion {name!r}; have {list(_FUSION_REGISTRY)}")
    return _FUSION_REGISTRY[name](d=d, **kwargs)
