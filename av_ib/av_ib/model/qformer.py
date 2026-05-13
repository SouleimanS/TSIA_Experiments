"""Video and audio Q-Formers, the trainable bridge between frozen encoders
and the frozen LLM.

This follows the AVHBench Align-FT pipeline exactly:

  VideoQFormer
    inner_qformer:  BLIP-2-initialized BERT (12 layers) with 32 learnable
                    query tokens that cross-attend to EVA-ViT features
                    per frame.
    frame_pos:      Embedding(8, 768) added to per-frame query outputs.
    outer_qformer:  BERT-initialized 2-layer transformer that cross-attends
                    across frame*query tokens, with 32 learnable query
                    tokens, producing a single (B, 32, 768) video summary.
    llama_proj:     Linear 768 -> LLM hidden (4096 for LLaMA-2-7B).

  AudioQFormer
    audio_pos:      Embedding(8, 1024) added to ImageBind clip features.
    qformer:        AVHBench-initialized 2-layer BERT with 8 query tokens.
    llama_proj:     Linear 768 -> 4096.

Both classes are fully trainable.

Implementation note: we DON'T reimplement the BERT-with-cross-attention.
We reuse `Blip2Base.init_video_Qformer(...)` from AVHBench's code, which
returns (Qformer_model, learnable_query_tokens). That helper builds a
HuggingFace BertModel with the `add_cross_attention=True` flag and
returns it alongside an `nn.Parameter` of shape (1, num_query, 768).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor


# Reuse the sys.path setup from encoders.py
_AVHBENCH_ROOT = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVHBench" / "AVHBench-Align-FT"
if str(_AVHBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_AVHBENCH_ROOT))

from video_llama.models.blip2 import Blip2Base  # noqa: E402


class VideoQFormer(nn.Module):
    """Video Q-Former: turns EVA-ViT per-frame features into a fixed set
    of LLM-ready tokens.

    Inputs:
        eva_features: (B*T, 257, 1408)   -- from VideoEncoder.forward
        num_frames:   T (so we can reshape internally)

    Output:
        (B, 32, llm_hidden)              -- 32 video tokens ready to be
                                             concatenated with audio and
                                             fed as inputs_embeds.
    """

    EVA_HIDDEN: int = 1408    # EVA-ViT-G hidden dim
    QFORMER_HIDDEN: int = 768  # BERT base hidden

    def __init__(
        self,
        num_query_tokens: int = 32,
        num_frames: int = 8,
        inner_num_layers: int = 12,
        outer_num_layers: int = 2,
        llm_hidden: int = 4096,
        blip2_ckpt_path: str = str(_AVHBENCH_ROOT / "models" / "blip2_pretrained_flant5xxl.pth"),
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.num_frames = num_frames

        # ---- Inner Q-Former: per-frame, 32 queries, BLIP-2 init ----
        # init_video_Qformer is a static method on Blip2Base that builds a
        # BERT with cross-attention enabled, plus learnable query tokens.
        # The naming is confusing -- it's a generic "Q-Former with cross-
        # attention to vision features", used for both image and video Q-
        # Formers in AVHBench.
        # init_Qformer returns a BLIP-2-init BERT with cross-attention.
        # Default depth is BERT-base's 12 layers; we override below if needed.
        self.inner_qformer, self.inner_query_tokens = Blip2Base.init_Qformer(
            num_query_token=num_query_tokens,
            vision_width=self.EVA_HIDDEN,
        )
        # Set the number of hidden layers AFTER construction by truncating.
        # BLIP-2's pretrained weights cover all 12 layers; if we ask for fewer,
        # we just slice the encoder's layer list.
        if inner_num_layers != 12:
            self.inner_qformer.bert.encoder.layer = self.inner_qformer.bert.encoder.layer[:inner_num_layers]
            self.inner_qformer.config.num_hidden_layers = inner_num_layers
        # Load BLIP-2 pretrained weights into the inner Q-Former.
        # Keys in the checkpoint are like "Qformer.bert.encoder.layer.0...";
        # we map them to "bert.encoder.layer.0..." by stripping the Qformer prefix.
        self._load_blip2_weights(self.inner_qformer, blip2_ckpt_path)

        # ---- Frame position embedding ----
        # 8 positions, projected to BERT hidden (768).
        self.frame_pos_embedding = nn.Embedding(num_frames, self.QFORMER_HIDDEN)

        # ---- Outer Q-Former: across-frame, 32 queries, BERT init ----
        # Same builder, but vision_width=768 (the dim of the inner Q-Former output).
        self.outer_qformer, self.outer_query_tokens = Blip2Base.init_Qformer(
            num_query_token=num_query_tokens,
            vision_width=self.QFORMER_HIDDEN,
        )
        if outer_num_layers != 12:
            self.outer_qformer.bert.encoder.layer = self.outer_qformer.bert.encoder.layer[:outer_num_layers]
            self.outer_qformer.config.num_hidden_layers = outer_num_layers
        # Outer Q-Former is not pre-trained -- it starts from BERT base init.
        # AVHBench's _v2 checkpoint contains weights for it but we use BLIP-2
        # for both (cleanest baseline for the v1 variant).

        # ---- Final projection to LLM hidden ----
        self.llama_proj = nn.Linear(self.QFORMER_HIDDEN, llm_hidden)

    @staticmethod
    def _load_blip2_weights(qformer: nn.Module, blip2_ckpt_path: str) -> None:
        """Load BLIP-2 pretrained Q-Former weights, stripping the 'Qformer.' prefix.

        BLIP-2 checkpoint keys: 'Qformer.bert.encoder.layer.0.attention...'
        Our Q-Former keys:      'bert.encoder.layer.0.attention...'
        """
        ckpt = torch.load(blip2_ckpt_path, map_location="cpu")
        # BLIP-2 ckpt is stored under 'model' top-level key in some versions
        sd = ckpt.get("model", ckpt)
        # Filter to Q-Former-only keys, strip the 'Qformer.' prefix
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("Qformer."):
                new_sd[k[len("Qformer."):]] = v
        missing, unexpected = qformer.load_state_dict(new_sd, strict=False)
        # Cross-attention layers won't be in the BLIP-2 ckpt (they're added
        # by our cross-attention=True flag). They'll show up in `missing` --
        # that's expected and fine, they keep their random init.
        n_cross = sum(1 for k in missing if "crossattention" in k)
        n_other_missing = len(missing) - n_cross
        print(f"  Inner Q-Former BLIP-2 load: "
              f"{n_cross} cross-attn keys randomly initialized, "
              f"{n_other_missing} other missing, {len(unexpected)} unexpected.")

    def forward(self, eva_features: Tensor) -> Tensor:
        """eva_features: (B*T, 257, 1408)  ->  (B, num_query, llm_hidden)"""
        bt, n_tok, d = eva_features.shape
        assert bt % self.num_frames == 0, (
            f"EVA features have B*T={bt} but num_frames={self.num_frames}"
        )
        b = bt // self.num_frames
        t = self.num_frames

        # --- Inner Q-Former: per-frame ---
        # Expand queries to batch
        inner_queries = self.inner_query_tokens.expand(bt, -1, -1)  # (B*T, 32, 768)
        # Cross-attend to EVA features
        encoder_atts = torch.ones(eva_features.shape[:-1], dtype=torch.long, device=eva_features.device)
        inner_out = self.inner_qformer.bert(
            query_embeds=inner_queries,
            encoder_hidden_states=eva_features,
            encoder_attention_mask=encoder_atts,
            return_dict=True,
        )
        # (B*T, 32, 768)
        per_frame_tokens = inner_out.last_hidden_state

        # --- Add frame position embedding ---
        # Reshape to (B, T, 32, 768), add (T, 768) position broadcast over query dim.
        per_frame_tokens = per_frame_tokens.reshape(b, t, self.num_query_tokens, self.QFORMER_HIDDEN)
        pos = self.frame_pos_embedding(
            torch.arange(t, device=per_frame_tokens.device)
        )  # (T, 768)
        per_frame_tokens = per_frame_tokens + pos.unsqueeze(0).unsqueeze(2)  # broadcast over B and query dim

        # Flatten frame and query dims: (B, T*32, 768)
        frame_seq = per_frame_tokens.reshape(b, t * self.num_query_tokens, self.QFORMER_HIDDEN)

        # --- Outer Q-Former: across-frame ---
        outer_queries = self.outer_query_tokens.expand(b, -1, -1)  # (B, 32, 768)
        outer_atts = torch.ones(frame_seq.shape[:-1], dtype=torch.long, device=frame_seq.device)
        outer_out = self.outer_qformer.bert(
            query_embeds=outer_queries,
            encoder_hidden_states=frame_seq,
            encoder_attention_mask=outer_atts,
            return_dict=True,
        )
        video_tokens = outer_out.last_hidden_state  # (B, 32, 768)

        # --- Project to LLM hidden ---
        return self.llama_proj(video_tokens)  # (B, 32, 4096)


class AudioQFormer(nn.Module):
    """Audio Q-Former: turns ImageBind clip features into LLM-ready tokens.

    Inputs:
        imagebind_features: (B, 8, 1024)

    Output:
        (B, 8, llm_hidden)
    """

    IMAGEBIND_HIDDEN: int = 1024
    QFORMER_HIDDEN: int = 768

    def __init__(
        self,
        num_query_tokens: int = 8,
        num_clips: int = 8,
        num_layers: int = 2,
        llm_hidden: int = 4096,
        audio_ckpt_path: str = str(_AVHBENCH_ROOT / "models" / "finetune_vicuna7b_videobranch.pth"),
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.num_clips = num_clips

        # Position embedding for the 8 clips, in ImageBind's hidden dim (1024)
        # so it's added BEFORE the Q-Former cross-attention.
        self.audio_position_embedding = nn.Embedding(num_clips, self.IMAGEBIND_HIDDEN)

        # The Q-Former itself
        self.qformer, self.query_tokens = Blip2Base.init_Qformer(
            num_query_token=num_query_tokens,
            vision_width=self.IMAGEBIND_HIDDEN,
        )
        if num_layers != 12:
            self.qformer.bert.encoder.layer = self.qformer.bert.encoder.layer[:num_layers]
            self.qformer.config.num_hidden_layers = num_layers

        # Projection to LLM hidden
        self.llama_proj = nn.Linear(self.QFORMER_HIDDEN, llm_hidden)

        # Load AVHBench's audio Q-Former checkpoint
        self._load_avhbench_weights(audio_ckpt_path)

    def _load_avhbench_weights(self, ckpt_path: str) -> None:
        """Load weights from AVHBench's finetune_vicuna7b_videobranch.pth
        which (despite the misleading name) contains the audio Q-Former.

        Relevant keys:
          'audio_Qformer.bert.*'           -> self.qformer.bert.*
          'audio_query_tokens'             -> self.query_tokens
          'audio_position_embedding.weight'-> self.audio_position_embedding.weight
          'audio_llama_proj.weight/bias'   -> self.llama_proj.weight/bias
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("model", ckpt)

        # Map AVHBench keys to our keys
        loaded = []
        # 1. Q-Former
        qformer_sd = {}
        for k, v in sd.items():
            if k.startswith("audio_Qformer."):
                qformer_sd[k[len("audio_Qformer."):]] = v
        missing, unexpected = self.qformer.load_state_dict(qformer_sd, strict=False)
        loaded.append(f"qformer: {len(qformer_sd)} keys loaded, "
                      f"{len(missing)} missing, {len(unexpected)} unexpected")

        # 2. Query tokens
        if "audio_query_tokens" in sd:
            with torch.no_grad():
                self.query_tokens.copy_(sd["audio_query_tokens"])
            loaded.append("query_tokens loaded")

        # 3. Position embedding
        if "audio_position_embedding.weight" in sd:
            with torch.no_grad():
                self.audio_position_embedding.weight.copy_(sd["audio_position_embedding.weight"])
            loaded.append("audio_position_embedding loaded")

        # 4. Llama projection
        if "audio_llama_proj.weight" in sd:
            with torch.no_grad():
                self.llama_proj.weight.copy_(sd["audio_llama_proj.weight"])
                if "audio_llama_proj.bias" in sd:
                    self.llama_proj.bias.copy_(sd["audio_llama_proj.bias"])
            loaded.append("llama_proj loaded")
        for line in loaded:
            print(f"  Audio Q-Former: {line}")

    def forward(self, imagebind_features: Tensor) -> Tensor:
        """imagebind_features: (B, 8, 1024)  ->  (B, num_query, llm_hidden)"""
        b, n_clips, d = imagebind_features.shape
        assert n_clips == self.num_clips, (
            f"AudioQFormer expects {self.num_clips} clips, got {n_clips}"
        )

        # Add position embedding
        pos_ids = torch.arange(n_clips, device=imagebind_features.device)
        pos = self.audio_position_embedding(pos_ids)  # (n_clips, 1024)
        audio_features = imagebind_features + pos.unsqueeze(0)  # (B, n_clips, 1024)

        # Cross-attend with Q-Former
        queries = self.query_tokens.expand(b, -1, -1)  # (B, 8, 768)
        atts = torch.ones(audio_features.shape[:-1], dtype=torch.long, device=audio_features.device)
        out = self.qformer.bert(
            query_embeds=queries,
            encoder_hidden_states=audio_features,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        audio_tokens = out.last_hidden_state  # (B, 8, 768)

        # Project to LLM hidden
        return self.llama_proj(audio_tokens)  # (B, 8, 4096)


def count_trainable(module: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total
