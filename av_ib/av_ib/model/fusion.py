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

    Two independent cross-attention blocks running in parallel:
        video' = block_v(video, audio)
        audio' = block_a(audio, video)
    """

    def __init__(self, d_model: int = 4096, n_heads: int = 8, ffn_mult: int = 1):
        super().__init__()
        self.block_v = _CrossAttnBlock(d_model, n_heads, ffn_mult)
        self.block_a = _CrossAttnBlock(d_model, n_heads, ffn_mult)

    def forward(self, video_tokens: Tensor, audio_tokens: Tensor):
        new_video = self.block_v(video_tokens, audio_tokens)
        new_audio = self.block_a(audio_tokens, video_tokens)
        return new_video, new_audio