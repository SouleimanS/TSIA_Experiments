"""Vicuna-7B-v0 wrapper with optional LoRA.

This is the LLM at the end of the pipeline. Everything upstream
(encoders, Q-Formers, fusion, bottleneck) produces a sequence of soft-
prompt tokens in the LLM's embedding space. The LLM consumes that
sequence plus the user's text question and produces an answer.

Responsibilities:
    - Load Vicuna-7B-v0 (frozen) + tokenizer.
    - Optionally wrap with PEFT LoRA (r=16, q/k/v/o, alpha=16).
    - Build the Vicuna-style prompt: '### Human: <text> ### Assistant: <answer>'.
    - Two forward modes:
        forward_train(av_tokens, prompt_text, answer_text)  -> loss
        forward_generate(av_tokens, prompt_text)             -> generated text

Constants:
    HIDDEN_SIZE = 4096   (Vicuna/LLaMA-1 7B hidden dim, matches our Q-Former
                          projection output)
    PROMPT_TEMPLATE = '### Human: {prompt} ### Assistant: '

How the AV tokens enter the LLM:
    We DON'T turn them into text tokens. We pass them as `inputs_embeds`
    by concatenating with the embedded text tokens. The LLM's forward
    accepts `inputs_embeds` instead of `input_ids` when the embedding
    layer has already been applied externally. This is the standard way
    of injecting visual/audio features into LLaMA-style models.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import torch
from torch import nn, Tensor
from transformers import LlamaForCausalLM, LlamaTokenizer


# Defaults
VICUNA_PATH = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVHBench" / "AVHBench-Align-FT" / "models" / "vicuna-7b-v0"


class LLMWrapper(nn.Module):
    """Vicuna-7B-v0 with optional LoRA.

    Forward signatures:
        forward_train(av_tokens, prompts, answers)
            av_tokens: (B, N_av, hidden)   -- 40 = 32 video + 8 audio AV tokens
            prompts:   List[str] of length B  -- e.g. 'Is the dog visible in the video?'
            answers:   List[str] of length B  -- e.g. 'Yes.'
            returns:   loss (scalar)

        forward_generate(av_tokens, prompts, max_new_tokens=10)
            av_tokens: (B, N_av, hidden)
            prompts:   List[str] of length B
            returns:   List[str] of length B  -- generated text per example
    """

    HIDDEN_SIZE: int = 4096

    # Vicuna-v0's expected format (matches AVHBench's conversation_video.py)
    PROMPT_TEMPLATE: str = "### Human: {prompt} ### Assistant: "

    def __init__(
        self,
        model_path: str = str(VICUNA_PATH),
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        precision: str = "fp16",
    ):
        super().__init__()
        self.use_lora = use_lora
        self.precision = precision

        # Tokenizer: slow tokenizer (use_fast=False) matches AVHBench's setup
        # and avoids a known tokenization quirk on some Vicuna checkpoints.
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        # Vicuna doesn't ship with a pad token. Pad with EOS for batching.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the frozen base model in the requested precision.
        dtype = torch.float16 if precision == "fp16" else torch.float32
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        # Freeze all base weights.
        for p in self.model.parameters():
            p.requires_grad = False

        if use_lora:
            # Wrap with PEFT LoRA. The wrapped model exposes the same API
            # (forward, generate) but with LoRA adapters injected at the
            # target modules. Only LoRA params are trainable.
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            # PEFT prints the trainable percentage by default but only at
            # construction; replicate it explicitly here for sanity.
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  LLM wrapped with LoRA: trainable={trainable/1e6:.1f}M / "
                  f"total={total/1e6:.1f}M ({100 * trainable / total:.3f}%)")
        else:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  LLM loaded, fully frozen: trainable={trainable} / total={total/1e6:.1f}M")

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def get_input_embeddings(self):
        """PEFT-friendly accessor for the token embedding layer.
        Works whether the model is PEFT-wrapped or not.
        """
        return self.model.get_input_embeddings()

    def _embed_text(self, text: str, device: torch.device, add_special_tokens: bool = False) -> Tensor:
        """Tokenize text and embed it. Returns (L, hidden) in model dtype.

        We always disable special tokens here because we control the
        prompt manually with BOS/EOS at the boundaries.
        """
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids.to(device)
        return self.get_input_embeddings()(ids).squeeze(0)  # (L, hidden)

    # ----------------------------------------------------------------------
    # Training forward
    # ----------------------------------------------------------------------

    def forward_train(
        self,
        av_tokens: Tensor,          # (B, N_av, hidden)
        prompts: List[str],         # B prompts
        answers: List[str],         # B answers
    ) -> Tensor:
        """Compute cross-entropy loss on the answer span.

        Sequence layout per example:
            [BOS] <split_left>      <av_tokens>      <split_right + answer>
                  '### Human: '     (32+8 embeds)    'question ### Assistant: answer'

        Loss is computed only on the answer tokens. Prefix positions get
        label=-100 (ignored by cross-entropy).
        """
        device = av_tokens.device
        b, n_av, _ = av_tokens.shape
        assert len(prompts) == b and len(answers) == b, (
            f"Batch size mismatch: av={b}, prompts={len(prompts)}, answers={len(answers)}"
        )

        # We split the Vicuna prompt around the AV tokens. The video/audio
        # tokens replace the '<AV>' placeholder in the conventional template:
        #   '### Human: <AV>question ### Assistant: '
        # Left segment: '### Human: ' (ends with a space; the AV tokens fit
        #                 right after a space which is how AVHBench does it)
        # Right segment: 'question ### Assistant: '
        # Answer segment: 'answer'

        left_text = "### Human: "
        # AVHBench wraps with "Close your eyes/ears" framing too; for v1
        # we keep it minimal. We can iterate on the template later.

        all_inputs = []      # list of (L_i, hidden) tensors
        all_labels = []      # list of (L_i,) tensors with -100 on prefix

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        embed = self.get_input_embeddings()

        for i in range(b):
            right_text = f"{prompts[i]} ### Assistant: "
            ans_text = answers[i]

            # Embed left segment (no BOS yet; we add it explicitly).
            left_ids = self.tokenizer(left_text, add_special_tokens=False,
                                       return_tensors="pt").input_ids.to(device)
            left_emb = embed(left_ids).squeeze(0)                  # (L_left, hidden)

            right_ids = self.tokenizer(right_text, add_special_tokens=False,
                                        return_tensors="pt").input_ids.to(device)
            right_emb = embed(right_ids).squeeze(0)                # (L_right, hidden)

            # Answer: include EOS at the end so the model learns to stop.
            ans_ids = self.tokenizer(ans_text, add_special_tokens=False,
                                      return_tensors="pt").input_ids.to(device).squeeze(0)
            ans_ids = torch.cat([ans_ids, torch.tensor([eos], device=device)])
            ans_emb = embed(ans_ids)                                # (L_ans, hidden)

            # BOS as a single embedded token.
            bos_emb = embed(torch.tensor([[bos]], device=device)).squeeze(0)

            # Concatenate: [BOS, left, AV, right, answer]
            full = torch.cat([bos_emb, left_emb, av_tokens[i], right_emb, ans_emb], dim=0)
            all_inputs.append(full)

            # Labels: -100 everywhere except on the answer tokens. The
            # model is trained next-token: position t predicts token t+1.
            # So the labels at positions [bos, left, av, right] should
            # be -100, and labels at positions of the answer tokens
            # should be the answer token ids themselves. Actually: the
            # label at position k is the token expected at position k+1
            # (HF's LlamaForCausalLM handles the shift internally), so
            # we just put -100 for prefix positions and ans_ids for the
            # answer positions (and -100 for the last position if we
            # want; HF shifts and drops anyway).
            n_prefix = bos_emb.size(0) + left_emb.size(0) + av_tokens.size(1) + right_emb.size(0)
            labels = torch.full((full.size(0),), -100, dtype=torch.long, device=device)
            labels[n_prefix : n_prefix + ans_ids.size(0)] = ans_ids
            all_labels.append(labels)

        # Pad to the longest sequence in the batch.
        max_len = max(x.size(0) for x in all_inputs)
        pad_emb = embed(torch.tensor([[self.tokenizer.pad_token_id]], device=device)).squeeze(0)

        padded_inputs = []
        padded_labels = []
        attn_masks = []
        for emb_seq, lab_seq in zip(all_inputs, all_labels):
            n = emb_seq.size(0)
            pad_n = max_len - n
            if pad_n > 0:
                emb_seq = torch.cat([emb_seq, pad_emb.expand(pad_n, -1)], dim=0)
                lab_seq = torch.cat([lab_seq, torch.full((pad_n,), -100, dtype=torch.long, device=device)])
            padded_inputs.append(emb_seq)
            padded_labels.append(lab_seq)
            mask = torch.ones(max_len, dtype=torch.long, device=device)
            mask[n:] = 0
            attn_masks.append(mask)

        inputs_embeds = torch.stack(padded_inputs, dim=0)            # (B, L, hidden)
        labels = torch.stack(padded_labels, dim=0)                    # (B, L)
        attention_mask = torch.stack(attn_masks, dim=0)               # (B, L)

        # Match model dtype.
        inputs_embeds = inputs_embeds.to(self.model.dtype)

        # Forward through LLaMA. The model handles the next-token shift
        # internally and returns cross-entropy averaged over non-ignored
        # positions.
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss

    # ----------------------------------------------------------------------
    # Generation
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def forward_generate(
        self,
        av_tokens: Tensor,          # (B, N_av, hidden)
        prompts: List[str],         # B prompts
        max_new_tokens: int = 10,
        do_sample: bool = False,
    ) -> List[str]:
        device = av_tokens.device
        b = av_tokens.size(0)
        assert len(prompts) == b

        embed = self.get_input_embeddings()
        bos = self.tokenizer.bos_token_id

        # Build the input embedding sequence per example, no answer span.
        all_inputs = []
        for i in range(b):
            left_text = "### Human: "
            right_text = f"{prompts[i]} ### Assistant: "
            bos_emb = embed(torch.tensor([[bos]], device=device)).squeeze(0)
            left_ids = self.tokenizer(left_text, add_special_tokens=False,
                                       return_tensors="pt").input_ids.to(device)
            left_emb = embed(left_ids).squeeze(0)
            right_ids = self.tokenizer(right_text, add_special_tokens=False,
                                        return_tensors="pt").input_ids.to(device)
            right_emb = embed(right_ids).squeeze(0)
            full = torch.cat([bos_emb, left_emb, av_tokens[i], right_emb], dim=0)
            all_inputs.append(full)

        # Pad to max len for batching.
        max_len = max(x.size(0) for x in all_inputs)
        pad_emb = embed(torch.tensor([[self.tokenizer.pad_token_id]], device=device)).squeeze(0)

        padded = []
        masks = []
        for emb_seq in all_inputs:
            n = emb_seq.size(0)
            pad_n = max_len - n
            if pad_n > 0:
                # Left-pad for generation (HF generate's default expectation).
                emb_seq = torch.cat([pad_emb.expand(pad_n, -1), emb_seq], dim=0)
                mask = torch.cat([
                    torch.zeros(pad_n, dtype=torch.long, device=device),
                    torch.ones(n, dtype=torch.long, device=device),
                ])
            else:
                mask = torch.ones(n, dtype=torch.long, device=device)
            padded.append(emb_seq)
            masks.append(mask)

        inputs_embeds = torch.stack(padded, dim=0).to(self.model.dtype)
        attention_mask = torch.stack(masks, dim=0)

        out_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in out_ids]
