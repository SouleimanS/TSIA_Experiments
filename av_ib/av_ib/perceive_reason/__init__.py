"""Perceive-then-reason: two-pass AV model with cross-modal attention.

Pass 1 (frozen, no grad): runs Qwen on the video to produce a text description
of the scene. The description is cached by video path (LRU, max 4096 entries).

Pass 2 (trained): runs the standard VIB pipeline (SinkAwareVIB on video,
NormTopKSinkVIB or standard VIB on audio), then feeds the AV tokens and the
description embeddings through a CrossModalAttention module before splicing
into the LLM. Only the cross_attn module, LoRA weights, and VIB weights are
trained.
"""
