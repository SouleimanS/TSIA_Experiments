"""Risk-coverage evaluation for abstention-trained models.

Metric: for each corruption type and strength, measure:
  - coverage  = fraction of questions the model chose to answer (not "I can't tell")
  - risk      = error rate on the answered subset only

A good fail-safe model drops coverage (abstains more) under corruptions of its
essential modality while keeping risk low on what it does answer.
Vanilla / C1 (no bottleneck) should keep coverage ~100% and let risk explode.
C2 (bottleneck) should drop coverage and hold risk — the crossing point is the result.

Scoring: teacher-forced log-probabilities of three responses under the abstain prompt:
  p_yes   = -NLL(answer="yes")
  p_no    = -NLL(answer="no")
  p_cant  = -NLL(answer="I can't tell")
Then:
  prediction = argmax{yes, no, cant}
  abstained  = prediction == "cant"
  correct    = prediction matches gold AND not abstained

Usage:
    python -m av_ib.eval.eval_riskcoverage \\
        --ann-path $ANN --video-root $VROOT \\
        --variant b_video_only \\
        [--ckpt-path runs/t2_nolora_b7e6/final.pt] \\
        --num-samples 150 --seed 99 \\
        --out-json runs/riskcoverage/c2a.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch


ABSTAIN_STR = "I can't tell"
ABSTAIN_PROMPT_SUFFIX = (
    " Answer with one word from the allowed vocabulary, "
    "or say \"I can't tell\" if the audio or video does not support an answer."
)

# Train corruptions (seen during abstention training)
TRAIN_CORRS = ["identity", "vid_zero", "vid_mean"]
# Held-out corruptions (never seen during training — the generalization test)
HELDOUT_CORRS = ["vid_noise", "vid_scale"]
# Negative control (audio degraded — model should NOT abstain on MUSIC-AVQA)
NEG_CORRS = ["aud_zero", "aud_mean"]

ALL_CORRS = TRAIN_CORRS + HELDOUT_CORRS + NEG_CORRS


def _norm(s: str) -> str:
    cleaned = "".join(c for c in s.strip().lower() if c.isalnum() or c.isspace())
    parts = cleaned.split()
    return parts[0] if parts else ""


def _score_response(model, inputs, response_str: str, prompt_len: int) -> float:
    """Teacher-forced NLL of response_str given the already-prepared inputs."""
    tok = model.qwen.tokenizer
    resp_ids = tok(response_str, add_special_tokens=False, return_tensors="pt").input_ids
    # Append response tokens to input_ids
    full_ids = torch.cat([inputs["input_ids"], resp_ids.to(inputs["input_ids"].device)], dim=1)
    labels = full_ids.clone()
    labels[:, :inputs["input_ids"].shape[1]] = -100  # mask prompt
    inp2 = {**inputs, "input_ids": full_ids}
    if "attention_mask" in inp2:
        inp2["attention_mask"] = torch.ones_like(full_ids)
    out = model.qwen.model.thinker(**inp2, labels=labels, use_audio_in_video=True)
    return float(out.loss.item())


def run_one(model, rec, corr_mode: str, gold: str) -> dict:
    """Run a single (question, corruption) pair. Returns prediction dict."""
    from av_ib.eval.explain import make_ablation

    v, a, p = [rec["video_path"]], [rec["audio_path"]], [rec["prompt"]]
    prompt_with_abstain = rec["prompt"] + ABSTAIN_PROMPT_SUFFIX

    model.z_ablation = make_ablation(corr_mode) if corr_mode != "identity" else None

    provider, _, _, _ = model._make_provider()
    model.qwen._current_provider = provider

    try:
        inputs, prompt_len = model.qwen._prep_inputs(v[0], prompt_with_abstain, answer=None)
        with torch.no_grad():
            nll_yes   = _score_response(model, inputs, "yes",        prompt_len)
            nll_no    = _score_response(model, inputs, "no",         prompt_len)
            nll_cant  = _score_response(model, inputs, ABSTAIN_STR,  prompt_len)
    finally:
        model.qwen._current_provider = None
        model.z_ablation = None

    scores = {"yes": -nll_yes, "no": -nll_no, "cant": -nll_cant}
    pred = max(scores, key=scores.__getitem__)
    abstained = (pred == "cant")
    answered_correct = (not abstained) and (_norm(pred) == _norm(gold))

    return {
        "pred": pred,
        "abstained": abstained,
        "answered_correct": answered_correct,
        "nll_yes": nll_yes,
        "nll_no": nll_no,
        "nll_cant": nll_cant,
    }


def main():
    ap = argparse.ArgumentParser(description="Risk-coverage eval for abstention models")
    ap.add_argument("--ann-path", required=True)
    ap.add_argument("--dataset", choices=["musicavqa", "avqa", "avhbench"],
                    default="musicavqa")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--variant", default="b_video_only")
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--num-samples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    from av_ib.model.av_model_v6 import AVModelV6


    tag = "untrained" if not args.ckpt_path else args.ckpt_path
    print(f"Building AVModelV6 (variant={args.variant}, ckpt={tag})...", flush=True)
    model = AVModelV6(use_lora=bool(args.ckpt_path), variant=args.variant).eval()
    model.set_sample_noise(False)
    if args.ckpt_path:
        sd = torch.load(args.ckpt_path, map_location="cpu")
        if "trainable_state" in sd:
            sd = sd["trainable_state"]
        own = dict(model.named_parameters())
        n_ok = sum(1 for k, v in sd.items() if k in own and not own[k].data.copy_(v.data) is None)
        print(f"  loaded {n_ok}/{len(sd)} params", flush=True)

    if args.dataset == "avqa":
        from av_ib.data.avqa import AVQADataset
        ds = AVQADataset(args.ann_path, args.video_root)
    elif args.dataset == "avhbench":
        from av_ib.data.avhbench_qa import AVHBenchQADataset
        ds = AVHBenchQADataset(args.ann_path, args.video_root, split="eval")
    else:
        from av_ib.data.musicavqa import MusicAVQADataset
        ds = MusicAVQADataset(args.ann_path, args.video_root)
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(len(ds)), min(args.num_samples, len(ds))))
    print(f"Evaluating {len(idxs)} samples × {len(ALL_CORRS)} corruptions\n", flush=True)

    # results[corr] = list of run_one dicts
    results = {c: [] for c in ALL_CORRS}

    for s, i in enumerate(idxs):
        rec = ds[i]
        gold = rec["answer"]
        for corr in ALL_CORRS:
            try:
                r = run_one(model, rec, corr, gold)
            except Exception as e:
                print(f"  [{s}] {corr}: skip: {e}", flush=True)
                r = {"pred": "err", "abstained": False, "answered_correct": False,
                     "nll_yes": float("nan"), "nll_no": float("nan"), "nll_cant": float("nan")}
            results[corr].append(r)

        if (s + 1) % args.every == 0:
            ref = results["identity"]
            cov = 1.0 - sum(r["abstained"] for r in ref) / max(len(ref), 1)
            risk = 1.0 - sum(r["answered_correct"] for r in ref) / max(
                sum(not r["abstained"] for r in ref), 1)
            print(f"  [{s+1}/{len(idxs)}] identity: cov={cov:.2f} risk={risk:.2f}", flush=True)

    # ── aggregate ──
    def agg(rs):
        n = len(rs)
        n_abstained = sum(r["abstained"] for r in rs)
        n_answered = n - n_abstained
        n_correct = sum(r["answered_correct"] for r in rs)
        coverage = n_answered / max(n, 1)
        risk = 1.0 - n_correct / max(n_answered, 1)
        return {"n": n, "coverage": coverage, "risk": risk,
                "n_abstained": n_abstained, "n_answered": n_answered, "n_correct": n_correct}

    summary = {corr: agg(results[corr]) for corr in ALL_CORRS}

    print("\n" + "=" * 70)
    print("RISK-COVERAGE REPORT")
    print("=" * 70)
    for group, corrs, label in [
        ("TRAIN corrs (seen)",   TRAIN_CORRS,   "expected: cov↓ under vid_*"),
        ("HELD-OUT corrs",       HELDOUT_CORRS, "the generalization test"),
        ("NEGATIVE control",     NEG_CORRS,     "expected: cov stays high"),
    ]:
        print(f"\n  {group}  [{label}]")
        print(f"  {'corruption':<14} {'coverage':>9} {'risk':>8} {'abstained':>10}")
        for c in corrs:
            a = summary[c]
            print(f"  {c:<14} {a['coverage']:>8.1%} {a['risk']:>8.1%} {a['n_abstained']:>10}/{a['n']}")
    print("=" * 70, flush=True)

    out = {"args": vars(args), "summary": summary,
           "per_sample": {c: results[c] for c in ALL_CORRS}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
