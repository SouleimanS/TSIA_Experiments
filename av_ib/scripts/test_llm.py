"""Smoke test for av_ib.model.llm."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from av_ib.model.llm import LLMWrapper


def main():
    print("=== Loading LLMWrapper (Vicuna-7B-v0 + LoRA) ===")
    llm = LLMWrapper(use_lora=True).cuda()

    # Fake AV tokens: 1 example, 40 AV tokens (32 video + 8 audio), 4096 hidden.
    av_tokens = torch.randn(1, 40, 4096, device="cuda")
    prompts = ["Is the dog visible in the video?"]

    # --- Test 1: forward_generate ---
    print("\n--- Generation test (greedy, max_new_tokens=10) ---")
    out = llm.forward_generate(av_tokens, prompts, max_new_tokens=10)
    print(f"  output: {out[0]!r}")

    # --- Test 2: forward_train (loss must be a scalar tensor with grad) ---
    print("\n--- Training forward test ---")
    answers = ["Yes."]
    loss = llm.forward_train(av_tokens, prompts, answers)
    print(f"  loss: {loss.item():.4f}")
    assert loss.requires_grad, "Loss must have grad enabled"
    loss.backward()
    print("  backward OK")

    # Confirm grads only on LoRA + nowhere else.
    n_lora_with_grad = 0
    n_lora_without_grad = 0
    for name, p in llm.named_parameters():
        if "lora_" in name:
            if p.grad is not None:
                n_lora_with_grad += 1
            else:
                n_lora_without_grad += 1
    print(f"  LoRA params with grad: {n_lora_with_grad}")
    print(f"  LoRA params without grad: {n_lora_without_grad}")
    assert n_lora_with_grad > 0, "Expected at least some LoRA params to have grad"
    print("  OK")

    print("\nLLM smoke test passed.")


if __name__ == "__main__":
    main()
