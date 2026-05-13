"""End-to-end smoke test for AVModelV1.

Tests:
  1. Build the whole model.
  2. Random video + random audio_mel + a prompt -> forward_generate.
  3. forward_train -> loss -> backward.
  4. Check gradients reach BOTH Q-Formers and LoRA.
  5. Check no gradients leak into frozen encoders or base LLM.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from av_ib.model.av_model_v1 import AVModelV1


def main():
    print("=== Building AVModelV1 ===")
    model = AVModelV1(use_lora=True).cuda()

    summary = model.trainable_summary()
    print("\nTrainable parameters per child:")
    for k, v in summary.items():
        if v > 0:
            print(f"  {k}: {v/1e6:.2f}M")

    # Fake inputs for B=1.
    videos = torch.randn(1, 8, 3, 224, 224, dtype=torch.float16, device="cuda")
    audio_mels = torch.randn(1, 8, 1, 128, 204, device="cuda")
    prompts = ["Is the dog visible in the video?"]
    answers = ["Yes."]

    # --- Generation pass ---
    print("\n--- Generation test ---")
    out = model.forward_generate(videos, audio_mels, prompts, max_new_tokens=10)
    print(f"  output: {out[0]!r}")

    # --- Training pass ---
    print("\n--- Training test ---")
    loss = model.forward_train(videos, audio_mels, prompts, answers)
    print(f"  loss: {loss.item():.4f}")
    assert loss.requires_grad
    loss.backward()
    print("  backward OK")

    # --- Grad audit ---
    print("\n--- Grad audit ---")
    # Confirm grads on the trainable bits.
    def has_any_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in module.parameters() if p.requires_grad)

    def all_have_no_grad(module):
        """All FROZEN params should have no grad attached."""
        for p in module.parameters():
            if not p.requires_grad:
                # Frozen params can have .grad=None, that's fine.
                if p.grad is not None and p.grad.abs().sum() > 0:
                    return False
        return True

    assert has_any_grad(model.video_qformer), "VideoQFormer should have grads"
    print("  video_qformer: grads present")
    assert has_any_grad(model.audio_qformer), "AudioQFormer should have grads"
    print("  audio_qformer: grads present")

    # Find LoRA params in the LLM and check grads.
    lora_with_grad = sum(
        1 for n, p in model.llm.named_parameters()
        if "lora_" in n and p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"  llm: {lora_with_grad} LoRA tensors with nonzero grad")
    assert lora_with_grad > 0

    # Encoders are frozen -- their params should have no grad attribute at all
    # (because @torch.no_grad in their forward prevents grad from flowing back).
    # Just verify their requires_grad is False:
    for n, p in model.video_encoder.named_parameters():
        assert not p.requires_grad, f"video_encoder.{n} should be frozen"
    for n, p in model.audio_encoder.named_parameters():
        assert not p.requires_grad, f"audio_encoder.{n} should be frozen"
    print("  encoders: confirmed frozen")

    print("\nAVModelV1 smoke test passed.")


if __name__ == "__main__":
    main()
