"""AVModelPerceive: perceive-then-reason two-pass model.

Inherits from AVModelV6. Adds:
  1. A perception pass (frozen, no grad) that generates a text description
     of the video via Qwen3-Omni before the main training/generation forward.
  2. A CrossModalAttention module that lets the VIB-processed AV tokens attend
     to the frozen description embeddings before splicing into the LLM.

Only trained parameters: cross_attn, LoRA weights, VIB weights (same set as
the base model). The perception pass (description generation + embedding) is
always wrapped in torch.no_grad().

Batch-size-1 assumption: same as the rest of the pipeline (CMIBSplicer).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from av_ib.model.av_model_v6 import AVModelV6
from av_ib.perceive_reason.cross_modal import CrossModalAttention

_DESC_CACHE_MAX = 4096

_PERCEPTION_PROMPT = "Briefly list the main objects and scene visible in this video."


class AVModelPerceive(AVModelV6):
    """Two-pass AV model: perceive (frozen) then reason (trained cross-attn).

    Args:
        n_perceive_frames: reserved for future use (currently the perception
            pass uses the video as-is via forward_generate; kept as a
            documented constructor arg for the training driver).
        cross_attn_heads:   number of heads in CrossModalAttention.
        cross_attn_dropout: dropout in CrossModalAttention (train-time only).
        All remaining kwargs are forwarded unchanged to AVModelV6.
    """

    def __init__(
        self,
        *args,
        n_perceive_frames: int = 4,
        cross_attn_heads: int = 8,
        cross_attn_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.n_perceive_frames = n_perceive_frames

        self.cross_attn = CrossModalAttention(
            d_model=self.D_MODEL,
            n_heads=cross_attn_heads,
            dropout=cross_attn_dropout,
        )

        # Description cache: video_path -> description string (LRU by insertion)
        self._desc_cache: dict[str, str] = {}
        self._desc_cache_keys: list[str] = []   # ordered by insertion for eviction

        # Per-forward stash; set before forward_train/generate, cleared after
        self._current_desc: Optional[str] = None

    # ------------------------------------------------------------------
    # Perception pass
    # ------------------------------------------------------------------

    def _generate_description(self, video_path: str) -> str:
        """Run one generate call to get a text description of the video.

        Called inside torch.no_grad() by _perceive_video.
        Uses the parent's forward_generate with no VIB provider so the base
        Qwen output is used for perception (the IB may distort perception).
        """
        # Temporarily disable any provider so we get raw Qwen output
        saved_provider = self.qwen._current_provider
        self.qwen._current_provider = None
        try:
            outs = self.qwen.forward_generate(
                videos=[video_path],
                audios=[video_path],
                prompts=[_PERCEPTION_PROMPT],
                max_new_tokens=80,
            )
        finally:
            self.qwen._current_provider = saved_provider
        return outs[0] if outs else ""

    def _perceive_video(self, video_path: str) -> str:
        """Return a cached or freshly generated text description of the video.

        The generation is wrapped in torch.no_grad() so no gradients flow
        through the perception pass.
        """
        if video_path in self._desc_cache:
            return self._desc_cache[video_path]

        with torch.no_grad():
            desc = self._generate_description(video_path)

        # Evict oldest entry if at capacity
        if len(self._desc_cache_keys) >= _DESC_CACHE_MAX:
            oldest = self._desc_cache_keys.pop(0)
            self._desc_cache.pop(oldest, None)

        self._desc_cache[video_path] = desc
        self._desc_cache_keys.append(video_path)
        return desc

    # ------------------------------------------------------------------
    # Description -> embeddings
    # ------------------------------------------------------------------

    def _desc_to_embeds(self, desc: str) -> Tensor:
        """Tokenize description and embed it with the frozen text embedder.

        Returns:
            (1, L, D) float32 tensor on the same device as the embedding table.
        """
        tok = self.qwen.tokenizer
        embed_layer = self.qwen.thinker_text_model.get_input_embeddings()
        dev = embed_layer.weight.device

        with torch.no_grad():
            ids = tok(
                desc,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(dev)
            # (1, L, D), returned as float32
            embeds = embed_layer(ids).float()

        return embeds  # (1, L, D)

    # ------------------------------------------------------------------
    # Provider override: attach CrossModalAttention to VIB output
    # ------------------------------------------------------------------

    def _make_provider(self):
        base_provider, kls, zs, masks = super()._make_provider()

        def perceive_provider(audio_out: Tensor, video_out: Tensor) -> Tensor:
            # Run base VIB + optional fusion
            z_j = base_provider(audio_out, video_out)   # (1, N_v+N_a, D)

            desc = self._current_desc
            if desc is None:
                # Skip cross-attn if no description was stashed (e.g. safety)
                return z_j

            # Build context from description embeddings
            ctx = self._desc_to_embeds(desc)            # (1, L, D) fp32
            # Cast to z_j's device and dtype before cross_attn
            ctx = ctx.to(device=z_j.device, dtype=z_j.dtype)

            # Ensure cross_attn module is on the same device as z_j
            cross_dev = next(self.cross_attn.parameters()).device
            if cross_dev != z_j.device:
                self.cross_attn.to(device=z_j.device)

            # Cross-attend in fp32 then cast back (avoids bf16 update loss)
            z_j = self.cross_attn(z_j.float(), ctx.float()).to(z_j.dtype)
            return z_j

        return perceive_provider, kls, zs, masks

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward_train(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        answers: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert len(videos) == 1, "AVModelPerceive assumes batch_size=1"
        self._current_desc = self._perceive_video(videos[0])
        try:
            return super().forward_train(videos, audios, prompts, answers)
        finally:
            self._current_desc = None

    # ------------------------------------------------------------------
    # Generation forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_generate(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        max_new_tokens: int = 20,
    ) -> List[str]:
        outs: List[str] = []
        for v, a, pr in zip(videos, audios, prompts):
            self._current_desc = self._perceive_video(v)
            try:
                outs.extend(
                    super().forward_generate(
                        [v], [a], [pr],
                        max_new_tokens=max_new_tokens,
                    )
                )
            finally:
                self._current_desc = None
        return outs
