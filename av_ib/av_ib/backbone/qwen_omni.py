"""Qwen3-Omni Thinker wrapper.

Loads Qwen3-Omni-30B-A3B-Instruct, freezes base weights, optionally wraps
the Thinker's text-model Linear layers with LoRA. Talker and Code2Wav are
loaded but disabled and never called.

forward_train / forward_generate:
    - Per-record (batch size 1 internally, loop over external batch).
    - Inputs are file paths for media, strings for text.
    - Optional av_token_provider callback: if set, splicing.py will use
      it (via hooks registered elsewhere) to swap raw encoder outputs for
      VIB-processed tokens before they reach the Thinker. The wrapper
      itself doesn't manage hooks — that's CMIBSplicer's job. The wrapper
      just propagates the callback into the forward kwargs so splicer
      knows when to fire.

Resolved from inspect_qwen_omni.py (Phase 0):
    audio encoder: thinker.audio_tower  (Qwen3OmniMoeAudioEncoder, 648M, out=2048)
    vision encoder: thinker.visual      (Qwen3OmniMoeVisionEncoder, 539M, out=2048)
    Thinker text model: thinker.model   (30.2B MoE, hidden_size=2048)
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import nn, Tensor


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


def _build_conversation(video_path: str, prompt: str, answer: Optional[str] = None) -> list:
    """Qwen3-Omni chat schema with one user video+text turn.
    If `answer` is given, append an assistant turn (for training).
    """
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt + " Answer with one word from the allowed vocabulary."},
            ],
        },
    ]
    if answer is not None:
        conv.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        })
    return conv


class QwenOmniWrapper(nn.Module):
    """Wraps Qwen3-Omni for AV question answering with optional LoRA on the Thinker."""

    AUDIO_ENCODER_PATH: str = "thinker.audio_tower"
    VISUAL_ENCODER_PATH: str = "thinker.visual"
    THINKER_HIDDEN: int = 2048
    AUDIO_OUT_DIM: int = 2048
    VISION_OUT_DIM: int = 2048

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        precision: str = "bf16",
    ):
        super().__init__()
        self.use_lora = use_lora

        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # We never need speech output for AVQA; disabling Talker saves ~10GB VRAM.
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

        # Freeze all base weights
        for p in self.model.parameters():
            p.requires_grad = False

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "linear_fc1", "linear_fc2"],
                bias="none",
                # NOTE: deliberately no task_type — we call generate on the outer
                # multi-modal model, not the PEFT wrapper. task_type="CAUSAL_LM"
                # makes PEFT try to bind prepare_inputs_for_generation, which
                # the inner text model doesn't expose.
            )
            # Inject LoRA into Linear layers inside thinker (q/k/v/o + linear_fc1/2).
            # We pass the inner text model so adapters land on the transformer layers,
            # not on encoders or unrelated submodules.
            from peft import get_peft_model as _get_peft_model
            self.model.thinker.model = _get_peft_model(
                self.model.thinker.model, lora_cfg,
            )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        # Attach the C-MIB splicer. It registers hooks but does nothing
        # unless self._current_provider is set by forward_train/generate.
        from av_ib.backbone.splicing import CMIBSplicer
        self.splicer = CMIBSplicer(self)
        self.splicer.attach()
        self._current_provider = None  # default; provider set per-call

        mode = "LoRA" if use_lora else "frozen"
        print(f"  QwenOmni ({mode}): trainable={trainable/1e6:.1f}M / "
              f"total={total/1e9:.1f}B ({100 * trainable / total:.3f}%)")

    @property
    def audio_encoder(self) -> nn.Module:
        return _resolve(self.model, self.AUDIO_ENCODER_PATH)

    @property
    def visual_encoder(self) -> nn.Module:
        return _resolve(self.model, self.VISUAL_ENCODER_PATH)

    @property
    def thinker_text_model(self) -> nn.Module:
        return self.model.thinker.model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prep_inputs(self, video_path: str, prompt: str, answer: Optional[str]):
        """Build processor inputs for one record. Returns the dict ready for
        model.forward(**inputs) or model.generate(**inputs), plus the
        number of prompt tokens (so train can mask out the prefix in labels)."""
        from qwen_omni_utils import process_mm_info

        # Two passes: one without answer to measure prompt length, one with
        # answer (if given) for training. For generation we only do one.
        conv_prompt = _build_conversation(video_path, prompt, answer=None)
        text_prompt = self.processor.apply_chat_template(
            conv_prompt, add_generation_prompt=True, tokenize=False,
        )
        # prompt_len computed AFTER processor below using marker search

        if answer is not None:
            conv_full = _build_conversation(video_path, prompt, answer=answer)
            text = self.processor.apply_chat_template(
                conv_full, add_generation_prompt=False, tokenize=False,
            )
        else:
            text = text_prompt

        audios, images, videos = process_mm_info(conv_prompt, use_audio_in_video=True)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Find the END of the assistant-turn opener in the actual token sequence.
        # The marker is "<|im_start|>assistant\n" = tokens [151644, 77091, 198].
        # Everything up to and INCLUDING those 3 tokens is prompt; the answer
        # begins immediately after. Search for the LAST occurrence in case the
        # prompt itself contains other im_start sequences.
        ids = inputs["input_ids"][0].tolist()
        marker = [151644, 77091, 198]
        prompt_len = None
        for i in range(len(ids) - 3, -1, -1):
            if ids[i:i+3] == marker:
                prompt_len = i + 3
                break
        if prompt_len is None:
            # Fallback: mask everything to prevent training on garbage
            prompt_len = len(ids)
        return inputs, prompt_len

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward_train(
        self,
        videos: List[str],
        audios: List[str],          # currently same paths as videos (audio extracted from mp4)
        prompts: List[str],
        answers: List[str],
        *,
        av_token_provider: Optional[Callable] = None,
    ) -> Tensor:
        """Per-record NLL on the answer span, averaged over the batch.

        If `av_token_provider` is given, it is forwarded to the splicer (which
        must be attached externally via CMIBSplicer). The wrapper just stashes
        it on `self._current_provider` so the hook can read it.
        """
        assert len(videos) == len(audios) == len(prompts) == len(answers)
        # Splicer reads this attribute from inside the forward hook.
        # Cleared at end of method so providers don't leak across calls.
        self._current_provider = av_token_provider

        losses = []
        try:
            for video_path, prompt, answer in zip(videos, prompts, answers):
                inputs, prompt_len = self._prep_inputs(video_path, prompt, answer)
                input_ids = inputs["input_ids"]
                # labels: -100 on prompt prefix, real ids on answer span
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100
                out = self.model.thinker(**inputs, labels=labels, use_audio_in_video=True)
                losses.append(out.loss)
            return torch.stack(losses).mean()
        finally:
            self._current_provider = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_generate(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        max_new_tokens: int = 10,
        *,
        av_token_provider: Optional[Callable] = None,
    ) -> List[str]:
        assert len(videos) == len(audios) == len(prompts)
        self._current_provider = av_token_provider
        outs = []
        try:
            for video_path, prompt in zip(videos, prompts):
                inputs, _ = self._prep_inputs(video_path, prompt, answer=None)
                text_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_audio=False,
                    use_audio_in_video=True,
                )
                input_len = inputs["input_ids"].shape[1]
                gen_ids = text_ids[:, input_len:] if text_ids.dim() == 2 else text_ids[input_len:]
                outs.append(self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip())
            return outs
        finally:
            self._current_provider = None


def _resolve(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj
