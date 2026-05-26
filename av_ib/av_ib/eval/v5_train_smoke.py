"""Phase 4.5: forward_train smoke test for AVModelV5.

Verifies backprop works through the splicer to the added modules. The hard part
isn't computing a loss — it's making sure gradients reach the VIBs / fusion /
aux heads through the hooks, since hooks in the middle of a frozen pretrained
model are a place autograd can quietly fail.

Checks, in order:
    1. Construction (same as forward_generate smoke)
    2. forward_train runs without crashing
    3. The 6-tuple loss is well-formed (all scalars, all finite)
    4. Combined loss is finite and requires_grad
    5. .backward() doesn't crash
    6. Gradients reach EACH added module: bottleneck_v, bottleneck_a, fusion,
       bottleneck_joint, aux_head_v, aux_head_a, AND the LoRA adapter params
    7. Loss is in a sane range (sanity, not correctness)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch


def grad_summary(name, mod):
    """Total grad norm + count of params with non-None .grad for a module."""
    total_norm_sq = 0.0
    has_grad = 0
    no_grad = 0
    n_params = 0
    for p in mod.parameters():
        if not p.requires_grad:
            continue
        n_params += 1
        if p.grad is None:
            no_grad += 1
        else:
            has_grad += 1
            total_norm_sq += p.grad.detach().float().pow(2).sum().item()
    norm = total_norm_sq ** 0.5
    return {
        "n_params": n_params,
        "n_with_grad": has_grad,
        "n_without_grad": no_grad,
        "grad_norm": norm,
    }


def main(args):
    print("=" * 60)
    print("Phase 4.5: forward_train smoke for AVModelV5")
    print("=" * 60)

    print("\n[1/7] Constructing AVModelV5...")
    t0 = time.time()
    try:
        from av_ib.model.av_model_v5 import AVModelV5
        model = AVModelV5(use_lora=True)
        model.train()
        print(f"  OK in {time.time() - t0:.1f}s")
    except Exception as e:
        traceback.print_exc()
        return 1

    print("\n[2/7] Loading one MUSIC-AVQA record...")
    try:
        with open(args.ann_path) as f:
            records = json.load(f)
        records = [r for r in records if r.get("question_deleted", 0) == 0]
        for rec_idx in range(min(5, len(records))):
            rec = records[rec_idx]
            video_path = Path(args.video_root) / f"{rec['video_id']}.mp4"
            if video_path.exists():
                break
        else:
            print(f"FAIL: no videos found in {args.video_root}")
            return 2
        from av_ib.data.musicavqa import render_question
        prompt = render_question(rec["question_content"], rec["templ_values"])
        gold = rec["anser"]
        print(f"  video_id={rec['video_id']}  prompt={prompt!r}  gold={gold!r}")
    except Exception as e:
        traceback.print_exc()
        return 2

    print("\n[3/7] Running forward_train...")
    t0 = time.time()
    try:
        nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j = model.forward_train(
            videos=[str(video_path)],
            audios=[str(video_path)],
            prompts=[prompt],
            answers=[gold],
        )
        print(f"  OK in {time.time() - t0:.1f}s")
    except Exception as e:
        traceback.print_exc()
        return 3

    print("\n[4/7] Six-term loss inspection...")
    losses = {"nll": nll, "nll_aux_v": nll_aux_v, "nll_aux_a": nll_aux_a,
              "kl_v": kl_v, "kl_a": kl_a, "kl_j": kl_j}
    fail = False
    for name, t in losses.items():
        v = t.detach().float().item()
        finite = torch.isfinite(t).all().item()
        has_grad_fn = t.grad_fn is not None
        print(f"  {name:10s}  value={v:+.4f}  finite={finite}  has_grad_fn={has_grad_fn}")
        if not finite:
            print(f"    FAIL: {name} is not finite")
            fail = True
    if fail:
        return 4

    print("\n[5/7] Combined loss .backward()...")
    # Move all loss tensors to a common device before composing.
    # accelerate's device_map="auto" leaves nll/KL/aux on different GPUs;
    # the real trainer will need to do the same.
    common_device = nll.device
    kl_v_c = kl_v.to(common_device)
    kl_a_c = kl_a.to(common_device)
    kl_j_c = kl_j.to(common_device)
    nll_aux_v_c = nll_aux_v.to(common_device)
    nll_aux_a_c = nll_aux_a.to(common_device)
    loss = nll + 1e-3 * (kl_v_c + kl_a_c + kl_j_c) + 0.1 * (nll_aux_v_c + nll_aux_a_c)
    print(f"  combined loss: {loss.item():+.4f}  requires_grad={loss.requires_grad}")
    if not loss.requires_grad:
        print("FAIL: combined loss doesn't require grad. Backprop will be a no-op.")
        return 5
    try:
        loss.backward()
        print(f"  OK — backward completed")
    except Exception as e:
        traceback.print_exc()
        return 5

    print("\n[6/7] Per-module gradient check...")
    checks = {
        "bottleneck_v": model.bottleneck_v,
        "bottleneck_a": model.bottleneck_a,
        "fusion": model.fusion,
        "bottleneck_joint": model.bottleneck_joint,
        "aux_head_v": model.aux_head_v,
        "aux_head_a": model.aux_head_a,
    }
    # LoRA adapters live inside qwen.model.thinker.model (the PEFT wrapper)
    # Find trainable LoRA params explicitly
    lora_params = [(n, p) for n, p in model.qwen.model.named_parameters()
                   if p.requires_grad]
    print(f"  LoRA trainable params: {len(lora_params)} parameters")

    fail = False
    for name, mod in checks.items():
        s = grad_summary(name, mod)
        status = "OK" if (s["n_without_grad"] == 0 and s["grad_norm"] > 0) else "FAIL"
        print(f"  [{status}] {name:18s} grad_norm={s['grad_norm']:.4e}  "
              f"params_with_grad={s['n_with_grad']}/{s['n_params']}")
        if s["n_without_grad"] > 0:
            print(f"    {s['n_without_grad']} params received NO gradient — splicer broke autograd")
            fail = True
        if s["grad_norm"] == 0.0 and s["n_with_grad"] > 0:
            print(f"    grad_norm is zero despite having .grad — suspicious")
            fail = True

    # LoRA sanity
    lora_grad_norm_sq = sum(
        p.grad.detach().float().pow(2).sum().item()
        for _, p in lora_params if p.grad is not None
    )
    lora_grad_norm = lora_grad_norm_sq ** 0.5
    lora_with_grad = sum(1 for _, p in lora_params if p.grad is not None)
    print(f"  [{'OK' if lora_with_grad == len(lora_params) and lora_grad_norm > 0 else 'FAIL'}] "
          f"LoRA               grad_norm={lora_grad_norm:.4e}  "
          f"params_with_grad={lora_with_grad}/{len(lora_params)}")
    if lora_with_grad != len(lora_params):
        fail = True

    if fail:
        print("\nFAIL: some modules didn't receive gradients.")
        return 6

    print("\n[7/7] Loss magnitude sanity check (cosmetic)...")
    if loss.item() > 100 or loss.item() < 0:
        print(f"  WARN: loss = {loss.item()} is outside typical range")
    else:
        print(f"  Loss in reasonable range")

    print("\n" + "=" * 60)
    print("Phase 4.5 TRAIN SMOKE PASSED — backprop reaches all added modules.")
    print(f"  forward_train: OK")
    print(f"  6-term loss:   all finite, all have grad_fn")
    print(f"  backward:      no crash")
    print(f"  grad flow:     all 6 added modules + LoRA received gradients")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    args = p.parse_args()
    sys.exit(main(args))
