"""Explainability instrument for AVModelV6 (see ROADMAP.md §4).

Subcommands (build out over time):
    e1   AV-reliance ablation  -- THE gate. Does the model use the AV stream?

E1 rationale
------------
The whole IB program assumes the LLM actually consumes the spliced audio/video
tokens z. If the answer is driven by the language prior (MUSIC-AVQA answers are
often guessable from the question text), then no bottleneck on z can change
anything, and tuning beta is meaningless. E1 measures this directly by
transforming z just before it enters the LLM and watching how much the answer
NLL and the generated answer change.

Conditions (all shape-preserving, applied to z_joint = [video ; audio]):
    identity     z unchanged (reference)
    zero         z := 0                      (AV embeddings removed)
    mean         each token := mean over tokens, per modality (per-token info gone)
    shuffle      tokens permuted along time, per modality (temporal alignment gone)
    video_only   audio slice zeroed          (keep video, drop audio)
    audio_only   video slice zeroed          (keep audio, drop video)

Read the verdict on `mean` and `zero`: if NLL barely rises and answers barely
flip, the model is prior-driven -> escalate (change task/benchmark) before
investing in the bottleneck.

Run on the UNTRAINED model (default, no checkpoint) to measure vanilla
Qwen3-Omni's AV reliance — the most fundamental, confound-free measurement.

Usage:
    python -m av_ib.eval.explain e1 \\
        --ann-path  $HOME/.../avqa-train.json \\
        --video-root $HOME/.../videos/all \\
        --num-samples 40 [--variant b_topk_nofusion] [--ckpt-path runs/.../final.pt] \\
        [--no-generate]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict

import torch


# ---------------------------------------------------------------------------
# z ablations: callables (z_joint, n_v, n_a) -> z_joint  (shape preserved)
# ---------------------------------------------------------------------------
def make_ablation(mode: str):
    def fn(z, n_v, n_a):
        zv, za = z[:, :n_v], z[:, n_v:]
        if mode == "identity":
            return z
        if mode == "zero":
            return torch.zeros_like(z)
        if mode == "mean":
            zv2 = zv.mean(dim=1, keepdim=True).expand_as(zv)
            za2 = za.mean(dim=1, keepdim=True).expand_as(za)
            return torch.cat([zv2, za2], dim=1)
        if mode == "shuffle":
            pv = torch.randperm(zv.size(1), device=z.device)
            pa = torch.randperm(za.size(1), device=z.device)
            return torch.cat([zv[:, pv], za[:, pa]], dim=1)
        if mode == "video_only":   # keep video, drop audio
            return torch.cat([zv, torch.zeros_like(za)], dim=1)
        if mode == "audio_only":   # keep audio, drop video
            return torch.cat([torch.zeros_like(zv), za], dim=1)
        if mode == "vid_zero":    # zero out only video tokens
            return torch.cat([torch.zeros_like(zv), za], dim=1)
        if mode == "aud_zero":    # zero out only audio tokens
            return torch.cat([zv, torch.zeros_like(za)], dim=1)
        if mode == "vid_mean":    # mean-replace only video tokens
            zv2 = zv.mean(dim=1, keepdim=True).expand_as(zv)
            return torch.cat([zv2, za], dim=1)
        if mode == "aud_mean":    # mean-replace only audio tokens
            za2 = za.mean(dim=1, keepdim=True).expand_as(za)
            return torch.cat([zv, za2], dim=1)
        if mode == "vid_noise":   # Gaussian noise on video (held-out test corruption)
            return torch.cat([zv + torch.randn_like(zv) * zv.std(), za], dim=1)
        if mode == "vid_scale":   # scale video to 50% (held-out test corruption)
            return torch.cat([zv * 0.5, za], dim=1)
        raise ValueError(f"unknown ablation mode {mode!r}")
    return fn


CONDITIONS = ["identity", "zero", "mean", "shuffle", "video_only", "audio_only"]


def _norm_answer(s: str) -> str:
    """Lowercase, strip punctuation, return the first word (the answer token)."""
    cleaned = "".join(c for c in s.strip().lower() if c.isalnum() or c.isspace())
    parts = cleaned.split()
    return parts[0] if parts else ""


def _type_key(meta: dict) -> str:
    t = meta.get("type") if isinstance(meta, dict) else None
    if t is None:
        return "unknown"
    try:
        parts = json.loads(t)
        return "/".join(str(p) for p in parts) if isinstance(parts, list) else str(t)
    except Exception:
        return str(t)


# ---------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------
def run_e1(args):
    from av_ib.model.av_model_v6 import AVModelV6
    from av_ib.data.musicavqa import MusicAVQADataset
    import random

    tag = "untrained" if not args.ckpt_path else f"ckpt={args.ckpt_path}"
    print(f"Building AVModelV6 (variant={args.variant}, {tag})...", flush=True)
    model = AVModelV6(use_lora=bool(args.ckpt_path), variant=args.variant).eval()
    model.set_sample_noise(False)  # deterministic z = mu
    if args.ckpt_path:
        sd = torch.load(args.ckpt_path, map_location="cpu")
        if "trainable_state" in sd:
            sd = sd["trainable_state"]
        own = dict(model.named_parameters())
        n_ok = 0
        for k, v in sd.items():
            if k in own:
                own[k].data.copy_(v.data)
                n_ok += 1
        print(f"  loaded {n_ok}/{len(sd)} params from checkpoint", flush=True)

    ds = MusicAVQADataset(args.ann_path, args.video_root)
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(len(ds)), min(args.num_samples, len(ds))))
    print(f"Probing {len(idxs)} samples; conditions={CONDITIONS}\n", flush=True)

    # nll[cond] = list of per-sample answer NLL; gen[cond] = list of generated answers
    nll = {c: [] for c in CONDITIONS}
    gen = {c: [] for c in CONDITIONS}
    golds, types = [], []
    # per-type delta accumulation (mean condition vs identity)
    per_type_dnll = defaultdict(list)

    for s, i in enumerate(idxs):
        rec = ds[i]
        v, a, p, ans = [rec["video_path"]], [rec["audio_path"]], [rec["prompt"]], [rec["answer"]]
        golds.append(_norm_answer(rec["answer"]))
        tkey = _type_key(rec.get("meta", {}))
        types.append(tkey)

        per_cond_nll = {}
        ok = True
        for c in CONDITIONS:
            model.z_ablation = make_ablation(c)
            try:
                with torch.no_grad():
                    out = model.forward_train(v, a, p, ans)
                this_nll = float(out[0].item())
            except Exception as e:
                print(f"  [{s}] {c}: NLL skip: {e}", flush=True)
                this_nll = float("nan"); ok = False
            nll[c].append(this_nll)
            per_cond_nll[c] = this_nll

            if args.generate:
                try:
                    with torch.no_grad():
                        g = model.forward_generate(v, a, p, max_new_tokens=5)[0]
                    gen[c].append(_norm_answer(g))
                except Exception as e:
                    print(f"  [{s}] {c}: gen skip: {e}", flush=True)
                    gen[c].append("")
        model.z_ablation = None

        if ok and per_cond_nll["identity"] == per_cond_nll["identity"]:
            per_type_dnll[tkey].append(per_cond_nll["mean"] - per_cond_nll["identity"])

        if (s + 1) % args.every == 0:
            base = per_cond_nll.get("identity", float("nan"))
            dz = per_cond_nll.get("zero", float("nan")) - base
            dm = per_cond_nll.get("mean", float("nan")) - base
            print(f"  [{s+1}/{len(idxs)}] nll_id={base:.3f}  "
                  f"dNLL(zero)={dz:+.3f}  dNLL(mean)={dm:+.3f}", flush=True)

    # ---- aggregate ----
    def m(lst):
        vals = [x for x in lst if x == x]
        return st.mean(vals) if vals else float("nan")

    base = m(nll["identity"])
    print("\n" + "=" * 66)
    print("E1 — AV-RELIANCE ABLATION")
    print("=" * 66)
    print(f"  samples: {len(idxs)}   baseline (identity) NLL: {base:.4f}\n")
    print(f"  {'condition':<12} {'mean_NLL':>10} {'dNLL':>9} {'dNLL%':>8} "
          f"{'ans_acc':>8} {'flip%':>7}")
    id_gen = gen["identity"] if args.generate else None
    for c in CONDITIONS:
        mc = m(nll[c])
        d = mc - base
        dp = 100.0 * d / base if base and base == base else float("nan")
        if args.generate:
            acc = 100.0 * sum(1 for g, gd in zip(gen[c], golds) if g == gd) / max(len(golds), 1)
            flip = 100.0 * sum(1 for g, gi in zip(gen[c], id_gen) if g != gi) / max(len(id_gen), 1)
        else:
            acc = flip = float("nan")
        print(f"  {c:<12} {mc:>10.4f} {d:>+9.4f} {dp:>+7.1f}% {acc:>7.1f}% {flip:>6.1f}%")

    # ---- per-type AV reliance (mean-ablation dNLL) ----
    if per_type_dnll:
        print("\n  Per-type dNLL (mean-replace vs identity); higher = more AV-reliant:")
        for tkey in sorted(per_type_dnll, key=lambda k: -m(per_type_dnll[k])):
            vals = per_type_dnll[tkey]
            print(f"    {tkey:<28} dNLL={m(vals):+.3f}  (n={len(vals)})")

    # ---- verdict ----
    d_mean = m(nll["mean"]) - base
    d_mean_pct = 100.0 * d_mean / base if base and base == base else 0.0
    flip_zero = (100.0 * sum(1 for g, gi in zip(gen["zero"], gen["identity"]) if g != gi)
                 / max(len(golds), 1)) if args.generate else float("nan")
    print("\n  " + "-" * 62)
    if d_mean_pct < 5:
        verdict = (f"AV-INERT — mean-replacing AV tokens changes NLL by only "
                   f"{d_mean_pct:.1f}%. The model is largely prior-driven on this "
                   f"data; a bottleneck on z cannot matter much. ESCALATE: pick an "
                   f"AV-dependent subset (see per-type table) or a harder benchmark.")
    elif d_mean_pct < 20:
        verdict = (f"WEAK reliance — NLL rises {d_mean_pct:.1f}% when AV per-token "
                   f"info is removed. The IB is testable but the effect ceiling is "
                   f"low; prefer the high-dNLL question types above.")
    else:
        verdict = (f"STRONG reliance — NLL rises {d_mean_pct:.1f}% without AV "
                   f"per-token info. The model uses the AV stream; the IB program "
                   f"is testable on this data.")
    print(f"  VERDICT: {verdict}")
    if args.generate and flip_zero == flip_zero:
        print(f"  (zeroing AV flips {flip_zero:.1f}% of generated answers)")
    print("=" * 66, flush=True)


# ---------------------------------------------------------------------------
# AVHARD — mine an "AV-essential" subset (questions that genuinely need BOTH
# modalities), defined behaviorally rather than by the dataset's type label.
# ---------------------------------------------------------------------------
def run_avhard(args):
    """For each candidate question, generate under identity / zero / video_only /
    audio_only on the untrained (vanilla) model and classify by modality need:

        AV-essential : id correct, audio_only WRONG, video_only WRONG
                       (and not zero-solvable) -> BOTH modalities required
        audio-suff   : id correct, audio_only correct  (video droppable)
        video-suff   : id correct, video_only correct  (audio droppable)
        prior-driven : zero correct                     (text prior solves it)
        too-hard     : id wrong

    Writes the AV-essential records (raw MUSIC-AVQA schema) to --out-json so the
    subset can be fed straight back as --ann-path for eval/training.
    """
    from av_ib.model.av_model_v6 import AVModelV6
    from av_ib.data.musicavqa import MusicAVQADataset
    import random

    tag = "untrained" if not args.ckpt_path else f"ckpt={args.ckpt_path}"
    print(f"Building AVModelV6 (variant={args.variant}, {tag}) for AV-hard mining...",
          flush=True)
    model = AVModelV6(use_lora=bool(args.ckpt_path), variant=args.variant).eval()
    model.set_sample_noise(False)
    if args.ckpt_path:
        sd = torch.load(args.ckpt_path, map_location="cpu")
        if "trainable_state" in sd:
            sd = sd["trainable_state"]
        own = dict(model.named_parameters())
        for k, v in sd.items():
            if k in own:
                own[k].data.copy_(v.data)

    ds = MusicAVQADataset(args.ann_path, args.video_root)
    # question_id -> raw record, for writing the output subset in source schema
    raw_by_qid = {r["question_id"]: r for r in ds.records}

    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(len(ds)), min(args.num_samples, len(ds))))
    # Mining conditions: only the four that determine modality need.
    conds = ["identity", "zero", "video_only", "audio_only"]
    print(f"Mining {len(idxs)} candidates; conditions={conds}\n", flush=True)

    buckets = {"av_essential": [], "audio_suff": [], "video_suff": [],
               "prior_driven": [], "too_hard": []}
    # per source-type tally of how many land in av_essential
    per_type = defaultdict(lambda: [0, 0])  # tkey -> [av_essential, total]

    for s, i in enumerate(idxs):
        rec = ds[i]
        v, a, p = [rec["video_path"]], [rec["audio_path"]], [rec["prompt"]]
        gold = _norm_answer(rec["answer"])
        meta = rec.get("meta", {})
        tkey = _type_key(meta)
        qid = meta.get("question_id")

        res = {}
        for c in conds:
            model.z_ablation = make_ablation(c)
            try:
                with torch.no_grad():
                    g = model.forward_generate(v, a, p, max_new_tokens=5)[0]
                res[c] = (_norm_answer(g) == gold)
            except Exception as e:
                print(f"  [{s}] {c}: gen skip: {e}", flush=True)
                res[c] = False
        model.z_ablation = None

        per_type[tkey][1] += 1
        if not res["identity"]:
            cat = "too_hard"
        elif res["zero"]:
            cat = "prior_driven"
        elif not res["video_only"] and not res["audio_only"]:
            cat = "av_essential"
            per_type[tkey][0] += 1
        elif res["audio_only"]:
            cat = "audio_suff"
        else:
            cat = "video_suff"
        buckets[cat].append(qid)

        if (s + 1) % args.every == 0:
            ne = len(buckets["av_essential"])
            print(f"  [{s+1}/{len(idxs)}] av_essential so far: {ne} "
                  f"({100.0*ne/(s+1):.1f}%)", flush=True)

    # ---- report ----
    n = len(idxs)
    print("\n" + "=" * 60)
    print("AV-HARD MINING — behavioral modality-dependence")
    print("=" * 60)
    for cat in ["av_essential", "audio_suff", "video_suff",
                "prior_driven", "too_hard"]:
        c = len(buckets[cat])
        print(f"  {cat:<14} {c:>5}  ({100.0*c/max(n,1):5.1f}%)")
    print("\n  AV-essential rate by source type (label vs reality):")
    for tkey in sorted(per_type, key=lambda k: -per_type[k][0]):
        ess, tot = per_type[tkey]
        print(f"    {tkey:<28} {ess:>4}/{tot:<4} ({100.0*ess/max(tot,1):5.1f}%)")

    # ---- write subset ----
    essential_qids = [q for q in buckets["av_essential"] if q in raw_by_qid]
    out_records = [raw_by_qid[q] for q in essential_qids]
    with open(args.out_json, "w") as f:
        json.dump(out_records, f, indent=2)
    print(f"\n  wrote {len(out_records)} AV-essential records -> {args.out_json}")
    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# ALABELS — emit EMPIRICAL abstention-training labels.
# A (question, corruption) pair earns an `abstain` target only if the model is
# RIGHT under identity and the corruption FLIPS it to wrong — a behavioral fact,
# not the circular "we removed evidence so abstain is correct". Questions whose
# answer survives a corruption keep the gold target under that corruption
# (the robustness / don't-over-abstain direction). Prior-driven questions (zero
# does not flip) therefore never get an abstain label, automatically.
# ---------------------------------------------------------------------------
def run_alabels(args):
    from av_ib.model.av_model_v6 import AVModelV6
    from av_ib.data.musicavqa import MusicAVQADataset
    import random

    tag = "untrained" if not args.ckpt_path else f"ckpt={args.ckpt_path}"
    print(f"Building AVModelV6 (variant={args.variant}, {tag}) for abstain labels...",
          flush=True)
    model = AVModelV6(use_lora=bool(args.ckpt_path), variant=args.variant).eval()
    model.set_sample_noise(False)
    if args.ckpt_path:
        sd = torch.load(args.ckpt_path, map_location="cpu")
        if "trainable_state" in sd:
            sd = sd["trainable_state"]
        own = dict(model.named_parameters())
        for k, v in sd.items():
            if k in own:
                own[k].data.copy_(v.data)

    ds = MusicAVQADataset(args.ann_path, args.video_root)
    raw_by_qid = {r["question_id"]: r for r in ds.records}
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(len(ds)), min(args.num_samples, len(ds))))

    # Per-modality corruptions: vid and aud variants separately.
    # Train corruptions are vid_zero/vid_mean. aud_zero/aud_mean are also labeled
    # (expected to be low-flip due to audio inertness) and serve as the negative
    # contrast: if a question flips under vid_zero but NOT aud_zero, that confirms
    # video-specific need. The training loop uses this contrast to teach sufficiency
    # reasoning vs artifact detection.
    corrs = ["vid_zero", "vid_mean", "aud_zero", "aud_mean"]
    conds = ["identity"] + corrs
    print(f"Labeling {len(idxs)} candidates; conditions={conds}\n", flush=True)

    out = []
    n_id_correct = 0
    flip_counts = {c: 0 for c in corrs}
    n_video_abstainable = 0  # flips under vid_zero OR vid_mean
    n_audio_abstainable = 0  # flips under aud_zero OR aud_mean
    n_contrast = 0           # flips video but NOT audio -> clean sufficiency signal

    for s, i in enumerate(idxs):
        rec = ds[i]
        v, a, p = [rec["video_path"]], [rec["audio_path"]], [rec["prompt"]]
        gold = _norm_answer(rec["answer"])
        qid = rec.get("meta", {}).get("question_id")

        preds = {}
        for c in conds:
            model.z_ablation = make_ablation(c)
            try:
                with torch.no_grad():
                    g = model.forward_generate(v, a, p, max_new_tokens=5)[0]
                preds[c] = (_norm_answer(g) == gold)
            except Exception as e:
                print(f"  [{s}] {c}: gen skip: {e}", flush=True)
                preds[c] = False
        model.z_ablation = None

        id_ok = preds["identity"]
        n_id_correct += int(id_ok)
        flips = {c: bool(id_ok and not preds[c]) for c in corrs}
        for c in corrs:
            flip_counts[c] += int(flips[c])

        vid_flips = flips["vid_zero"] or flips["vid_mean"]
        aud_flips = flips["aud_zero"] or flips["aud_mean"]
        if id_ok and vid_flips:
            n_video_abstainable += 1
        if id_ok and aud_flips:
            n_audio_abstainable += 1
        if id_ok and vid_flips and not aud_flips:
            n_contrast += 1

        if qid in raw_by_qid:
            entry = dict(raw_by_qid[qid])
            entry["identity_correct"] = id_ok
            entry["flips_vid_zero"] = flips["vid_zero"]
            entry["flips_vid_mean"] = flips["vid_mean"]
            entry["flips_aud_zero"] = flips["aud_zero"]
            entry["flips_aud_mean"] = flips["aud_mean"]
            out.append(entry)

        if (s + 1) % args.every == 0:
            print(f"  [{s+1}/{len(idxs)}] id_correct={n_id_correct} "
                  f"vid_abstainable={n_video_abstainable} "
                  f"contrast(vid_not_aud)={n_contrast}", flush=True)

    n = len(idxs)
    print("\n" + "=" * 60)
    print("ABSTAIN LABELS — per-modality empirical flip labels")
    print("=" * 60)
    print(f"  candidates           : {n}")
    print(f"  identity-correct     : {n_id_correct} ({100.0*n_id_correct/max(n,1):.1f}%)")
    for c in corrs:
        print(f"  flips under {c:<10}: {flip_counts[c]} "
              f"({100.0*flip_counts[c]/max(n,1):.1f}%)")
    print(f"  video-abstainable    : {n_video_abstainable} ({100.0*n_video_abstainable/max(n,1):.1f}%)")
    print(f"  audio-abstainable    : {n_audio_abstainable} ({100.0*n_audio_abstainable/max(n,1):.1f}%)")
    print(f"  contrast(vid not aud): {n_contrast} ({100.0*n_contrast/max(n,1):.1f}%)")

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {len(out)} labeled records -> {args.out_json}")
    print("  Fields: identity_correct, flips_vid_zero, flips_vid_mean,")
    print("          flips_aud_zero, flips_aud_mean")
    print("  Training: vid_flip->abstain / aud_flip->gold / no_flip->gold")
    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="AVModelV6 explainability instrument")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e1 = sub.add_parser("e1", help="AV-reliance ablation (the gate)")
    e1.add_argument("--ann-path", required=True)
    e1.add_argument("--video-root", required=True)
    e1.add_argument("--variant", default="b_topk_nofusion")
    e1.add_argument("--ckpt-path", default=None,
                    help="If set, load this checkpoint (and enable LoRA). "
                         "Default: untrained model = vanilla Qwen AV reliance.")
    e1.add_argument("--num-samples", type=int, default=40)
    e1.add_argument("--seed", type=int, default=42)
    e1.add_argument("--every", type=int, default=5)
    e1.add_argument("--no-generate", dest="generate", action="store_false", default=True,
                    help="Skip generation (NLL only) — faster.")
    e1.set_defaults(func=run_e1)

    mh = sub.add_parser("avhard", help="Mine an AV-essential subset (needs both modalities)")
    mh.add_argument("--ann-path", required=True)
    mh.add_argument("--video-root", required=True)
    mh.add_argument("--variant", default="b_topk_nofusion")
    mh.add_argument("--ckpt-path", default=None,
                    help="Default: untrained vanilla model (confound-free mining).")
    mh.add_argument("--num-samples", type=int, default=400,
                    help="Candidate pool to scan (the essential subset is a fraction).")
    mh.add_argument("--seed", type=int, default=7)
    mh.add_argument("--every", type=int, default=20)
    mh.add_argument("--out-json", default="runs/avhard_subset.json")
    mh.set_defaults(func=run_avhard)

    al = sub.add_parser("alabels", help="Emit empirical abstention-training labels")
    al.add_argument("--ann-path", required=True)
    al.add_argument("--video-root", required=True)
    al.add_argument("--variant", default="b_topk_nofusion")
    al.add_argument("--ckpt-path", default=None,
                    help="Default: untrained vanilla model (label with the model "
                         "we will actually fine-tune from).")
    al.add_argument("--num-samples", type=int, default=3000,
                    help="Candidates to scan; abstainable yield is a fraction.")
    al.add_argument("--seed", type=int, default=42,
                    help="Use the TRAIN seed so labels cover the training pool.")
    al.add_argument("--every", type=int, default=50)
    al.add_argument("--out-json", default="runs/abstain_labels.json")
    al.set_defaults(func=run_alabels)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
