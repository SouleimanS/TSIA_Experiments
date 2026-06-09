"""SFT abstention training for AVModelV6.

Trains the model with a teacher-forced NLL loss to prefer "I can't tell" under
video corruption while preserving answer quality on clean evidence.

Loss composition:
    abstain loss  : NLL("I can't tell" | corrupted_vid_input)   — flip pairs
    anchor loss   : NLL(gold | clean_input)                      — capability
    negctrl loss  : NLL(gold | aud_corrupted_input)              — negative ctrl

    total = lambda_abstain * L_abstain + lambda_anchor * L_anchor + lambda_negctrl * L_negctrl

Consuming exactly the same abstain_labels.json as the DPO path (dpo_data.py)
so the two training paths are a clean head-to-head controlled for supervision source.

Usage:
    python -m av_ib.train.train_sft_abstain \\
        --labels-json runs/abstain_labels.json \\
        --variant b_video_only \\
        --num-steps 400 --lr 5e-5 \\
        --log-path runs/sft/c2a_sft.jsonl \\
        --ckpt-path runs/sft/c2a_sft_final.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from av_ib.train.dpo_data import ABSTAIN_STR, ABSTAIN_PROMPT_SUFFIX


def _nll_response(model, video_path: str, prompt: str, response: str, corr: str) -> torch.Tensor:
    """Mean per-token NLL of `response` given (prompt + spliced AV tokens) under corr.

    Mirrors the teacher-forced scoring in eval_riskcoverage._score_response.
    Uses mean (not sum) so NLL is not sensitive to response length differences
    between "yes"/"no" and "I can't tell".
    """
    from av_ib.eval.explain import make_ablation

    tok = model.qwen.tokenizer
    model.z_ablation = make_ablation(corr) if corr != "identity" else None
    provider, _, _, _ = model._make_provider()
    model.qwen._current_provider = provider
    try:
        inputs, prompt_len = model.qwen._prep_inputs(video_path, prompt, answer=None)
        resp_ids = tok(response, add_special_tokens=False,
                       return_tensors="pt").input_ids.to(inputs["input_ids"].device)
        full_ids = torch.cat([inputs["input_ids"], resp_ids], dim=1)
        labels = full_ids.clone()
        labels[:, :inputs["input_ids"].shape[1]] = -100
        inp2 = {**inputs, "input_ids": full_ids}
        if "attention_mask" in inp2:
            inp2["attention_mask"] = torch.ones_like(full_ids)
        out = model.qwen.model.thinker(**inp2, labels=labels, use_audio_in_video=True)
        return out.loss  # already mean over non-masked tokens
    finally:
        model.qwen._current_provider = None
        model.z_ablation = None


def main(args):
    print("=" * 60)
    print(f"SFT abstention | variant={args.variant} | steps={args.num_steps} "
          f"| lam_abstain={args.lam_abstain} lam_anchor={args.lam_anchor}")
    print("=" * 60)

    from av_ib.model.av_model_v6 import AVModelV6
    from av_ib.train.dpo_data import PreferencePairDataset, _collate
    from av_ib.train.loop import trainable_params, trainable_state_dict

    model = AVModelV6(use_lora=args.use_lora, variant=args.variant)
    model.set_sample_noise(False)

    if args.ckpt_init:
        sd = torch.load(args.ckpt_init, map_location="cpu")
        sd = sd.get("trainable_state", sd)
        own = dict(model.named_parameters())
        n = sum(1 for k, v in sd.items() if k in own and own[k].data.copy_(v.data) is not None)
        print(f"  init from {args.ckpt_init}: loaded {n}/{len(sd)} params")

    if args.freeze_vib:
        for mod in (model.bottleneck_v, model.bottleneck_a):
            for p in mod.parameters():
                p.requires_grad = False
        print("  VIB frozen (vanilla+LoRA condition)")

    # Re-use the same PreferencePairDataset so data is identical to the DPO path.
    ds = PreferencePairDataset(
        args.labels_json,
        include_anchor=not args.no_anchor,
        include_negctrl=not args.no_negctrl,
        anchor_ratio=args.anchor_ratio,
    )
    print(f"  preference pairs: {len(ds)}  by kind: {ds.kind_counts}")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=_collate)

    opt = torch.optim.AdamW(trainable_params(model), lr=args.lr, weight_decay=0.05)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.ckpt_path:
        Path(args.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w")

    def cycle(ld):
        while True:
            for b in ld:
                yield b

    it = cycle(loader)
    model.train()
    t0 = time.time()
    n_abstain_correct = 0  # NLL(abstain) < NLL(gold) under corruption

    for step in range(args.num_steps):
        pair = next(it)
        vp, prompt, corr = pair["video_path"], pair["prompt"], pair["corr"]
        chosen, rejected = pair["chosen"], pair["rejected"]  # same fields as DPO
        kind = pair["kind"]

        # For "abstain" pairs: chosen=ABSTAIN_STR, rejected=gold.
        # We minimise NLL of the *chosen* response (the preferred one).
        # For "anchor" and "negctrl" pairs: chosen=gold — same treatment.
        if kind == "abstain":
            nll = _nll_response(model, vp, prompt, chosen, corr)
            loss = args.lam_abstain * nll
        elif kind == "anchor":
            nll = _nll_response(model, vp, prompt, chosen, corr)
            loss = args.lam_anchor * nll
        else:  # negctrl
            nll = _nll_response(model, vp, prompt, chosen, corr)
            loss = args.lam_negctrl * nll

        # For abstain pairs, also check if model already prefers abstain (no grad).
        if kind == "abstain":
            with torch.no_grad():
                nll_rj = _nll_response(model, vp, prompt, rejected, corr)
            n_abstain_correct += int(nll.item() < nll_rj.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable_params(model), 1.0)
        opt.step()

        rec = {
            "step": step, "kind": kind, "corr": corr,
            "loss": float(loss.item()), "nll_chosen": float(nll.item()),
            "grad_norm": float(gn), "elapsed_s": time.time() - t0,
        }
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if step % args.print_every == 0:
            absn = f" abstain_acc={n_abstain_correct/(step+1):.2f}" if kind == "abstain" else ""
            print(f"  step {step:4d} [{kind:8s}/{corr:8s}] "
                  f"loss={rec['loss']:.3f} nll_ch={rec['nll_chosen']:.3f} "
                  f"gn={gn:.2f}{absn} t={rec['elapsed_s']:.0f}s", flush=True)

        if args.save_every and args.ckpt_path and (step + 1) % args.save_every == 0:
            p = Path(args.ckpt_path).parent / f"step_{step+1}.pt"
            torch.save({"step": step, "trainable_state": trainable_state_dict(model)}, p)
            print(f"  [ckpt] {p.name}", flush=True)

    log_f.close()
    if args.ckpt_path:
        torch.save({"step": args.num_steps - 1,
                    "trainable_state": trainable_state_dict(model)}, args.ckpt_path)
        print(f"saved final -> {args.ckpt_path}")
    print(f"done: {args.num_steps} steps in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels-json", required=True)
    p.add_argument("--variant", default="b_video_only")
    p.add_argument("--use-lora", action="store_true", default=False)
    p.add_argument("--freeze-vib", action="store_true", default=False)
    p.add_argument("--ckpt-init", default=None)
    p.add_argument("--num-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lam-abstain", type=float, default=1.0,
                   help="weight on abstain-pair NLL loss")
    p.add_argument("--lam-anchor", type=float, default=0.5,
                   help="weight on capability-anchor NLL loss")
    p.add_argument("--lam-negctrl", type=float, default=0.5,
                   help="weight on negative-control NLL loss")
    p.add_argument("--anchor-ratio", type=float, default=1.0)
    p.add_argument("--no-anchor", action="store_true", default=False)
    p.add_argument("--no-negctrl", action="store_true", default=False)
    p.add_argument("--log-path", default="runs/sft/log.jsonl")
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)
    main(p.parse_args())
