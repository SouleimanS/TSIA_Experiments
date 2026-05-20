"""Fusion modules between Q-Former outputs and the LLM.

Two implementations:
    Identity              - no fusion, passes tokens through unchanged (used by v1)
    MutualCrossAttention  - video tokens attend to audio, audio attends to video,
                            via a single transformer block per modality.

Both expect:
    video_tokens: (B, Nv, D)   D = LLM hidden size (4096 for Vicuna-7B)
    audio_tokens: (B, Na, D)

Both return:
    video_tokens': (B, Nv, D)
    audio_tokens': (B, Na, D)
"""
from __future__ import annotations

import torch
from torch import nn, Tensor


class Identity(nn.Module):
    """No-op fusion. v1 uses this."""

    def forward(self, video_tokens: Tensor, audio_tokens: Tensor):
        return video_tokens, audio_tokens


class _CrossAttnBlock(nn.Module):
    """One block: cross-attn (q from x, k/v from y) + residual + FFN + residual.

    LayerNorms are pre-norm style.
    """

    def __init__(self, d_model: int = 4096, n_heads: int = 8, ffn_mult: int = 2):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        q = self.ln_q(x)
        kv = self.ln_kv(y)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln_ffn(x))
        return x


class MutualCrossAttention(nn.Module):
    """v2 fusion: each modality cross-attends to the other.

    With n_blocks=1 (default): one cross-attention block per direction,
    both run in parallel using the ORIGINAL counterpart as KV. This is
    the original v2 behavior, kept for backward compatibility.

    With n_blocks>1: stacks of independent blocks per direction. Each
    pair of blocks shares the parallel-update pattern: at layer i,
    video_{i+1} = block_v_i(video_i, audio_i), audio_{i+1} = block_a_i(audio_i, video_i).
    So the two streams update in lockstep, each layer seeing the
    other modality at the same depth.
    """

    def __init__(self, d_model: int = 4096, n_heads: int = 8, ffn_mult: int = 1, n_blocks: int = 1):
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        self.n_blocks = n_blocks
        self.blocks_v = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, ffn_mult) for _ in range(n_blocks)
        ])
        self.blocks_a = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, ffn_mult) for _ in range(n_blocks)
        ])

    def forward(self, video_tokens: Tensor, audio_tokens: Tensor):
        v, a = video_tokens, audio_tokens
        for block_v, block_a in zip(self.blocks_v, self.blocks_a):
            new_v = block_v(v, a)
            new_a = block_a(a, v)
            v, a = new_v, new_a
        return v, a