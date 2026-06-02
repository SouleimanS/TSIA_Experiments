"""Phase 0 reconnaissance: dump Qwen3-Omni's structure to locate hook points."""
from __future__ import annotations

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, AutoProcessor


MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def main():
    print(f"Loading {MODEL_PATH} ...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("\n=== TOP-LEVEL MODULES ===")
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters()) / 1e6
        print(f"  {name}: {type(child).__name__}  ({n_params:.1f}M params)")

    print("\n=== FULL MODULE TREE (depth 2) ===")
    for name, child in model.named_modules():
        depth = name.count(".")
        if depth <= 1 and name:
            n_params = sum(p.numel() for p in child.parameters()) / 1e6
            print(f"  {name}: {type(child).__name__}  ({n_params:.1f}M)")

    print("\n=== SEARCHING FOR ENCODER MODULES ===")
    for name, module in model.named_modules():
        lname = name.lower()
        if any(k in lname for k in ("audio_tower", "audio_encoder", "aut",
                                    "visual", "vision_tower", "vit", "talker")):
            n_params = sum(p.numel() for p in module.parameters()) / 1e6
            depth = name.count(".")
            if depth <= 3:  # avoid printing every leaf
                print(f"  {name}: {type(module).__name__}  ({n_params:.1f}M)")

    print("\n=== CANDIDATE LoRA TARGET MODULES (Linear in Thinker) ===")
    seen = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "thinker" in name.lower():
            seen.add(name.split(".")[-1])
    print(f"  Linear suffixes in Thinker: {sorted(seen)}")

    print("\n=== CONFIG SUMMARY ===")
    cfg = model.config
    print(f"  Model class: {type(model).__name__}")
    print(f"  Config class: {type(cfg).__name__}")
    # Try a few common attribute paths for hidden dim
    for attr in ("hidden_size", "thinker_config", "audio_config", "vision_config"):
        if hasattr(cfg, attr):
            val = getattr(cfg, attr)
            if hasattr(val, "hidden_size"):
                print(f"  cfg.{attr}.hidden_size = {val.hidden_size}")
            else:
                print(f"  cfg.{attr} = {val}")


if __name__ == "__main__":
    main()
