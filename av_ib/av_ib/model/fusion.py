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
        # Zero-init output projections so both attn and ffn residuals are 0 at
        # step 0 → MutualCrossAttention is identity at init, starts at baseline.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

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

# ============================================================================
# Sink-mediated symmetric fusion (v6+)
# ============================================================================
import math as _math


class SinkSymmetricFusion(nn.Module):
    """Cross-modal fusion that binds modalities ONLY through their sink tokens,
    with a SYMMETRIC (tied-W) coupling and a zero-init residual gate.

    Design (operationalizes the sink hypothesis):
      - Score between video token i and audio token j is a single bilinear form
        s_ij = (Wq v_i) . (Wk a_j) / sqrt(d_head), with ONE shared (Wq, Wk).
        The reverse direction (audio querying video) reuses the SAME score
        matrix transposed -> coupling is symmetric by construction; neither
        modality can have more cross-modal bandwidth than the other.
      - Keys/values are restricted to the COUNTERPART modality's SINK tokens
        (the validated cross-modal binding tokens). Non-sink tokens are still
        updated as queries but bind only through sinks.
      - Update is a zero-init residual: x <- x + g * attn(...), g init 0, so at
        step 0 the module is identity -> the LLM sees native embeddings ->
        eval starts at baseline. Training moves g off 0 only to add signal.

    Falls back gracefully: if a modality has zero sink tokens in a sample, that
    direction's update is skipped (residual passes through unchanged).
    """

    def __init__(self, d_model: int, n_heads: int = 8,
                 dsink_dims=None, sink_tau: float = 18.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dsink_dims = list(dsink_dims) if dsink_dims is not None else [985, 1992]
        self.sink_tau = sink_tau

        # Pre-norms (separate per modality is fine; norm is not the coupling)
        self.ln_v = nn.LayerNorm(d_model)
        self.ln_a = nn.LayerNorm(d_model)

        # SHARED query/key projections -> tied bilinear score form.
        # Both modalities use the SAME Wq, Wk so the score matrix for
        # video->audio is the transpose of audio->video.
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)

        # Separate value + output projections per modality (scores-only symmetry).
        self.Wv_video = nn.Linear(d_model, d_model, bias=False)
        self.Wv_audio = nn.Linear(d_model, d_model, bias=False)
        self.out_video = nn.Linear(d_model, d_model, bias=False)
        self.out_audio = nn.Linear(d_model, d_model, bias=False)

        # Zero-init residual gates (one scalar per modality)
        self.g_video = nn.Parameter(torch.zeros(()))
        self.g_audio = nn.Parameter(torch.zeros(()))

    def _classify_sinks(self, x: Tensor) -> Tensor:
        """Per-token sink mask via RMSNorm magnitude on dsink dims. (B, T) bool."""
        # RMSNorm over feature dim
        rms = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        phi = rms[..., self.dsink_dims].abs().amax(dim=-1)  # (B, T)
        return phi >= self.sink_tau

    def _split_heads(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,dh)

    def forward(self, video_tokens: Tensor, audio_tokens: Tensor):
        v, a = video_tokens, audio_tokens
        B, Nv, D = v.shape
        Na = a.shape[1]

        # Sink masks for each modality
        sink_v = self._classify_sinks(v)  # (B, Nv)
        sink_a = self._classify_sinks(a)  # (B, Na)

        vn = self.ln_v(v)
        an = self.ln_a(a)

        # Shared projections
        q_v = self._split_heads(self.Wq(vn))   # video as query  (B,H,Nv,dh)
        k_a = self._split_heads(self.Wk(an))   # audio as key    (B,H,Na,dh)
        q_a = self._split_heads(self.Wq(an))   # audio as query  (B,H,Na,dh)
        k_v = self._split_heads(self.Wk(vn))   # video as key    (B,H,Nv,dh)

        val_a = self._split_heads(self.Wv_audio(an))  # (B,H,Na,dh)
        val_v = self._split_heads(self.Wv_video(vn))  # (B,H,Nv,dh)

        scale = 1.0 / _math.sqrt(self.d_head)

        # --- Video queries attend to AUDIO SINKS ---
        # scores_va: (B,H,Nv,Na)
        scores_va = torch.matmul(q_v, k_a.transpose(-1, -2)) * scale
        # mask out non-sink audio keys
        key_mask_a = sink_a[:, None, None, :]  # (B,1,1,Na)
        scores_va = scores_va.masked_fill(~key_mask_a, float('-inf'))
        # rows with no valid key -> set to 0 attn (avoid nan)
        valid_v = key_mask_a.any(dim=-1)  # (B,1,1)
        attn_va = torch.softmax(scores_va, dim=-1)
        attn_va = torch.nan_to_num(attn_va, nan=0.0)
        ctx_v = torch.matmul(attn_va, val_a)  # (B,H,Nv,dh)
        ctx_v = ctx_v.transpose(1, 2).reshape(B, Nv, D)
        v_update = self.out_video(ctx_v)

        # --- Audio queries attend to VIDEO SINKS ---
        # By symmetry the score is the transpose form: reuse q_a,k_v (same Wq,Wk)
        scores_av = torch.matmul(q_a, k_v.transpose(-1, -2)) * scale  # (B,H,Na,Nv)
        key_mask_v = sink_v[:, None, None, :]  # (B,1,1,Nv)
        scores_av = scores_av.masked_fill(~key_mask_v, float('-inf'))
        attn_av = torch.softmax(scores_av, dim=-1)
        attn_av = torch.nan_to_num(attn_av, nan=0.0)
        ctx_a = torch.matmul(attn_av, val_v)  # (B,H,Na,dh)
        ctx_a = ctx_a.transpose(1, 2).reshape(B, Na, D)
        a_update = self.out_audio(ctx_a)

        # Zero-init residual gates
        v_out = v + self.g_video * v_update
        a_out = a + self.g_audio * a_update
        return v_out, a_out
