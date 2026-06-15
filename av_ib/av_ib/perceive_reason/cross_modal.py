"""CrossModalAttention: attend AV tokens to description embeddings.

Queries come from the concatenated video+audio tokens (B, N, D).
Keys and values come from the frozen description embeddings (B, M, D).
A residual connection is used with a LayerNorm; out_proj is zero-initialized
so an untrained model is the identity.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Cross-modal attention: AV tokens attend to description embeddings.

    Args:
        d_model: token/embedding dimension (2048 for Qwen3-Omni).
        n_heads: number of attention heads.
        dropout: dropout probability (applied during training only).
    """

    def __init__(self, d_model: int = 2048, n_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

        # Zero-init out_proj so untrained model = identity residual
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Cross-attend x (AV tokens) to context (description embeddings).

        Args:
            x:       (B, N, D) — audio+video tokens, queries.
            context: (B, M, D) — description embeddings, keys and values.

        Returns:
            (B, N, D) — attended tokens with residual connection.
        """
        residual = x
        q = self.q_proj(x)        # (B, N, D)
        k = self.k_proj(context)  # (B, M, D)
        v = self.v_proj(context)  # (B, M, D)

        B, N, D = q.shape
        M = k.shape[1]
        head_dim = D // self.n_heads

        # Reshape to (B, n_heads, seq, head_dim) — no in-place ops
        q = q.view(B, N, self.n_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(B, M, self.n_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(B, M, self.n_heads, head_dim).permute(0, 2, 1, 3)

        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        # (B, n_heads, N, head_dim) -> (B, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out)

        return self.norm(residual + out)
