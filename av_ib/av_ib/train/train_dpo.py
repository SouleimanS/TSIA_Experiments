"""DPO abstention training for AVModelV6 (FINER-inspired).

Trains the model to prefer "I can't tell" over a confident answer when the
essential (video) evidence is removed, while preferring the gold answer when
evidence is intact or only the non-essential modality is degraded. See
docs/abstention_dpo_plan.md.

Reference model = the same backbone with LoRA adapters disabled (PEFT
`disable_adapter`). VIB modules are shared and run under no_grad for the
reference pass (tiny module; acceptable KL anchor, as in TRL shared-base DPO).

Usage:
    python -m av_ib.train.train_dpo \\
        --labels-json runs/abstain_labels.json \\
        --variant b_video_only \\
        --num-steps 400 --beta-dpo 0.1 --lr 5e-5 \\
        --log-path runs/c2a_dpo/log.jsonl \\
        --ckpt-path runs/c2a_dpo/final.pt
    # C1-DPO (no bottleneck): add --no-bottleneck-grad is implicit via --variant b_video_only
    # but for the true vanilla+LoRA condition pass --use-lora and freeze VIB with --freeze-vib.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _seq_logprob(model, video_path: str, prompt: str, response: str,
                 corr: str, *, reference: bool) -> torch.Tensor:
    """Summed log-prob of `response` given (prompt + spliced AV tokens) under
    the given corruption. If reference=True, LoRA adapters are disabled.

    Mirrors the input prep used by eval_riskcoverage: build prompt-only inputs,
    append response token ids, score with the thinker, sum log-probs over the
    response span.
    """
    from av_ib.eval.explain import make_ablation

    tok = model.qwen.tokenizer
    model.z_ablation = make_ablation(corr) if corr != "identity" else None
    provider, _, _, _ = model._make_provider()
    model.qwen._current_provider = provider

    def _run():
        inputs, _ = model.qwen._prep_inputs(video_path, prompt, answer=None)
        resp_ids = tok(response, add_special_tokens=False,
                       return_tensors="pt").input_ids.to(inputs["input_ids"].device)
        n_prompt = inputs["input_ids"].shape[1]
        full_ids = torch.cat([inputs["input_ids"], resp_ids], dim=1)
        inp2 = {**inputs, "input_ids": full_ids}
        if "attention_mask" in inp2:
            inp2["attention_mask"] = torch.ones_like(full_ids)
        out = model.qwen.model.thinker(**inp2, use_audio_in_video=True)
        logits = out.logits  # (1, T, V)
        # logits at position t predict token t+1; response tokens occupy
        # [n_prompt, n_prompt + R). Their predictions come from logits[n_prompt-1 .. ].
        R = resp_ids.shape[1]
        pred_logits = logits[:, n_prompt - 1: n_prompt - 1 + R, :]
        logp = F.log_softmax(pred_logits.float(), dim=-1)
        # with device_map sharding, logits live on the last shard, not cuda:0
        idx = resp_ids.unsqueeze(-1).to(logp.device)
        tok_logp = logp.gather(-1, idx).squeeze(-1)  # (1, R)
        return tok_logp.sum()

    try:
        peft_model = model.qwen.model.thinker.model
        if reference:
            with torch.no_grad():
                if hasattr(peft_model, "disable_adapter"):
                    with peft_model.disable_adapter():
                        return _run().detach()
                return _run().detach()
        else:
            return _run()
    finally:
        model.qwen._current_provider = None
        model.z_ablation = None


def main(args):
    print("=" * 60)
    print(f"DPO abstention | variant={args.variant} | steps={args.num_steps} "
          f"| beta_dpo={args.beta_dpo}")
    print("=" * 60)

    from av_ib.model.av_model_v6 import AVModelV6
    from av_ib.train.dpo_data import PreferencePairDataset, _collate
    from av_ib.train.loop import trainable_params, trainable_state_dict

    model = AVModelV6(use_lora=args.use_lora, variant=args.variant)
    model.set_sample_noise(False)  # deterministic z for stable preference scoring

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

    if not args.use_lora and not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("nothing trainable: enable LoRA or unfreeze VIB")

    ds = PreferencePairDataset(
        args.labels_json,
        args.video_root,
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
    n_acc = 0  # running count where chosen preferred (margin > 0)

    for step in range(args.num_steps):
        pair = next(it)
        vp, prompt, corr = pair["video_path"], pair["prompt"], pair["corr"]
        chosen, rejected = pair["chosen"], pair["rejected"]

        # reference log-probs (no grad, adapters off)
        ref_ch = _seq_logprob(model, vp, prompt, chosen,   corr, reference=True)
        ref_rj = _seq_logprob(model, vp, prompt, rejected, corr, reference=True)
        # policy log-probs (grad on)
        pol_ch = _seq_logprob(model, vp, prompt, chosen,   corr, reference=False)
        pol_rj = _seq_logprob(model, vp, prompt, rejected, corr, reference=False)

        logits = args.beta_dpo * ((pol_ch - ref_ch) - (pol_rj - ref_rj))
        loss = -F.logsigmoid(logits)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable_params(model), 1.0)
        opt.step()

        margin = float((pol_ch - pol_rj).item())
        n_acc += int(margin > 0)
        rec = {
            "step": step, "kind": pair["kind"], "corr": corr,
            "loss": float(loss.item()), "margin": margin,
            "pol_ch": float(pol_ch.item()), "pol_rj": float(pol_rj.item()),
            "grad_norm": float(gn), "acc_running": n_acc / (step + 1),
            "elapsed_s": time.time() - t0,
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if step % args.print_every == 0:
            print(f"  step {step:4d} [{pair['kind']:8s}/{corr:8s}] "
                  f"loss={rec['loss']:.3f} margin={margin:+.2f} "
                  f"acc={rec['acc_running']:.2f} gn={gn:.2f} t={rec['elapsed_s']:.0f}s",
                  flush=True)

        if args.save_every and args.ckpt_path and (step + 1) % args.save_every == 0:
            p = Path(args.ckpt_path).parent / f"step_{step+1}.pt"
            torch.save({"step": step, "trainable_state": trainable_state_dict(model)}, p)
            print(f"  [ckpt] {p.name}", flush=True)

    log_f.close()
    if args.ckpt_path:
        torch.save({"step": args.num_steps - 1,
                    "trainable_state": trainable_state_dict(model)}, args.ckpt_path)
        print(f"saved final -> {args.ckpt_path}")
    print(f"done: {args.num_steps} steps in {time.time()-t0:.0f}s  "
          f"final pref-acc={n_acc/max(args.num_steps,1):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels-json", required=True)
    p.add_argument("--video-root", required=True,
                   help="Directory with <video_id>.mp4 files (to rebuild paths)")
    p.add_argument("--variant", default="b_video_only")
    p.add_argument("--use-lora", action="store_true", default=False)
    p.add_argument("--freeze-vib", action="store_true", default=False,
                   help="C1 condition: freeze VIB, train LoRA only")
    p.add_argument("--ckpt-init", default=None,
                   help="warm-start trainable params from an existing checkpoint")
    p.add_argument("--num-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta-dpo", type=float, default=0.1)
    p.add_argument("--anchor-ratio", type=float, default=1.0,
                   help="keep anchors up to anchor_ratio x #abstain pairs (<1 to balance)")
    p.add_argument("--no-anchor", action="store_true", default=False)
    p.add_argument("--no-negctrl", action="store_true", default=False)
    p.add_argument("--log-path", default="runs/dpo/log.jsonl")
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)
    main(p.parse_args())
