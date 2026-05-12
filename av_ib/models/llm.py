"""LLM backbone wrapper.

Contract:

    LLM.forward(inputs_embeds, labels=None) -> dict(logits, loss=None)

    inputs_embeds: (B, N, d) -- the bottlenecked tokens T act as the
                                soft-prompt that the LLM conditions on.
                                We feed them as `inputs_embeds` so we do
                                not have to project to vocab and back.
    labels:        (B, Ly)   -- answer token ids. When given, the wrapper
                                computes the answer-NLL (cross-entropy)
                                using teacher forcing on the answer ids,
                                following each example's prompt-conditioned
                                state.

The smoke-test stub is a 1-layer transformer with a small vocab head.
It is enough to verify the forward+backward through the whole pipeline
and to see that gradients reach Φ and the bottleneck.

Real backbones (Vicuna, LLaMA via HuggingFace) will be added via the
`hf_causal_lm` registry entry. The wrapper handles `frozen=True` so the
training loop never updates LLM parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass
class LLMOutput:
    logits: Tensor                       # (B, N + Ly, vocab) -- only answer slice contributes to loss
    loss: Optional[Tensor]               # scalar; None if labels not provided


class TinyLLMStub(nn.Module):
    """1-layer decoder-only transformer used for smoke testing.

    Inputs are taken via `inputs_embeds`. The wrapper appends embeddings
    of the answer tokens for teacher forcing, runs them through a small
    transformer, projects to vocab, and computes cross-entropy on the
    answer slice only (so the loss does not depend on the visual/audio
    tokens, only on whether they conditioned the prediction correctly).
    """

    def __init__(self, hidden_size: int = 4096, vocab_size: int = 32000, n_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size,
            batch_first=True, dropout=0.0,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        inputs_embeds: Tensor,           # (B, N, d)
        labels: Optional[Tensor] = None, # (B, Ly)
    ) -> LLMOutput:
        b, n, d = inputs_embeds.shape

        if labels is None:
            # Just run the prefix; useful for generation. Not exercised in smoke test.
            h = self.layer(inputs_embeds)
            logits = self.lm_head(h)
            return LLMOutput(logits=logits, loss=None)

        # Build embed sequence: [inputs_embeds ; embed(labels)].
        ans_embeds = self.token_emb(labels)                # (B, Ly, d)
        h = torch.cat([inputs_embeds, ans_embeds], dim=1)   # (B, N+Ly, d)

        # Causal mask over the full sequence so the answer slice attends
        # left-to-right and to all prefix tokens.
        total = h.size(1)
        causal_mask = torch.triu(
            torch.ones(total, total, device=h.device, dtype=torch.bool), diagonal=1,
        )
        h = self.layer(h, src_mask=causal_mask)
        logits = self.lm_head(h)                            # (B, N+Ly, vocab)

        # Loss: predict label[t] from position (N + t - 1). The position
        # right before the first answer token is N-1 (the last prefix token).
        ly = labels.size(1)
        # Shifted logits/labels for next-token prediction over the answer span.
        pred_logits = logits[:, n - 1 : n - 1 + ly, :]      # (B, Ly, vocab)
        loss = F.cross_entropy(
            pred_logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
        )
        return LLMOutput(logits=logits, loss=loss)


_LLM_REGISTRY = {
    "tiny_stub": TinyLLMStub,
    # 'hf_causal_lm': HFCausalLMWrapper,  # add when integrating Vicuna/LLaMA
}


def build_llm(name: str, hidden_size: int, **kwargs) -> nn.Module:
    if name not in _LLM_REGISTRY:
        raise KeyError(f"Unknown LLM {name!r}; have {list(_LLM_REGISTRY)}")
    return _LLM_REGISTRY[name](hidden_size=hidden_size, **kwargs)
