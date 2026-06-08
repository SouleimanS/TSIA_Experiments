"""Held-out accuracy evaluation for AVModelV6 checkpoints.

Excludes the training indices (--train-seed / --train-max-samples) so every
sample seen here is one the model has never trained on.

Usage (one run):
    python -m av_ib.eval.eval_v6_heldout \
        --ann-path  $ANN_PATH  --video-root $VIDEO_ROOT \
        --variant b_topk_nofusion \
        --label untrained \
        --num-samples 150 --seed 99 \
        [--ckpt-path runs/gate3_beta7e6/final.pt]

Re-run for each checkpoint; results accumulate under --out-dir and a
comparison table is printed when sibling metrics.json files exist.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_heldout(ann_path: str, video_root: str,
                   train_max: int, train_seed: int,
                   eval_n: int, eval_seed: int) -> tuple:
    """Return (dataset, heldout_indices)."""
    from av_ib.data.musicavqa import MusicAVQADataset
    ds = MusicAVQADataset(ann_path, video_root)
    all_idx = list(range(len(ds)))
    train_rng = random.Random(train_seed)
    train_set = set(train_rng.sample(all_idx, min(train_max, len(all_idx))))
    held = [i for i in all_idx if i not in train_set]
    eval_rng = random.Random(eval_seed)
    chosen = sorted(eval_rng.sample(held, min(eval_n, len(held))))
    print(f"Dataset: {len(ds)} total | train excluded: {len(train_set)} "
          f"| held-out pool: {len(held)} | evaluating: {len(chosen)}", flush=True)
    return ds, chosen


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(variant: str, ckpt_path: str | None):
    from av_ib.model.av_model_v6 import AVModelV6
    use_lora = ckpt_path is not None
    print(f"Building AVModelV6 (variant={variant}, "
          f"{'untrained' if not use_lora else ckpt_path})...", flush=True)
    model = AVModelV6(use_lora=use_lora, variant=variant).eval()
    model.set_sample_noise(False)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        if "trainable_state" in sd:
            sd = sd["trainable_state"]
        own = dict(model.named_parameters())
        n_ok = sum(
            1 for k, v in sd.items()
            if k in own and not own[k].data.copy_(v.data).isnan().any()
        )
        print(f"  Loaded {n_ok}/{len(sd)} params", flush=True)
    return model


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _run(model, ds, idxs: list[int], every: int = 10) -> list[dict]:
    from av_ib.eval.answer_parser import parse_answer
    results = []
    t0 = time.time()
    for s, i in enumerate(idxs):
        rec = ds[i]
        video_path = rec["video_path"]
        prompt     = rec["prompt"]
        gold       = rec["answer"].strip().lower()
        meta       = rec.get("meta", {})

        try:
            with torch.no_grad():
                raws = model.forward_generate(
                    [video_path], [video_path], [prompt], max_new_tokens=10
                )
            raw = raws[0]
        except Exception as e:
            raw = f"<ERROR: {e}>"

        parsed  = parse_answer(raw)
        correct = int(parsed == gold)

        type_str = meta.get("type", "[]") if isinstance(meta, dict) else "[]"
        try:
            parts    = json.loads(type_str)
            modality = str(parts[0]) if len(parts) > 0 else "?"
            subtype  = str(parts[1]) if len(parts) > 1 else "?"
        except Exception:
            modality, subtype = "?", "?"

        results.append({
            "idx":        i,
            "video_id":   meta.get("video_id", "") if isinstance(meta, dict) else "",
            "question":   prompt,
            "gold":       gold,
            "raw_pred":   raw,
            "pred":       parsed,
            "correct":    correct,
            "modality":   modality,
            "subtype":    subtype,
            "type_key":   f"{modality}/{subtype}",
        })

        if (s + 1) % every == 0:
            acc  = 100 * sum(r["correct"] for r in results) / len(results)
            rate = len(results) / (time.time() - t0)
            print(f"  [{s+1:>4}/{len(idxs)}]  acc={acc:.1f}%  "
                  f"rate={rate:.2f}/s", flush=True)

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    n_ok = sum(r["correct"] for r in results)

    def _acc(pairs):
        return {k: {"acc": v[0] / v[1] if v[1] else 0.0, "n": v[1]}
                for k, v in sorted(pairs.items())}

    by_mod  = defaultdict(lambda: [0, 0])
    by_sub  = defaultdict(lambda: [0, 0])
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        for d, key in [(by_mod, r["modality"]),
                       (by_sub, r["subtype"]),
                       (by_type, r["type_key"])]:
            d[key][0] += r["correct"]
            d[key][1] += 1

    return {
        "n":            n,
        "n_correct":    n_ok,
        "accuracy":     round(n_ok / n, 4),
        "per_modality": _acc(by_mod),
        "per_subtype":  _acc(by_sub),
        "per_type":     _acc(by_type),
    }


def _print_summary(m: dict, label: str) -> None:
    sep = "=" * 66
    print()
    print(sep)
    print(f"  Held-out eval · {label}")
    print(sep)
    print(f"  Overall: {m['accuracy']:.1%}  ({m['n_correct']}/{m['n']})")
    print()
    print("  ─── Per Modality ───")
    for mod, d in m["per_modality"].items():
        print(f"    {mod:<22}: {d['acc']:.1%}  (n={d['n']})")
    print()
    print("  ─── Per Type ───")
    for tk, d in sorted(m["per_type"].items(), key=lambda x: -x[1]["n"]):
        bar = "#" * int(d["acc"] * 20)
        print(f"    {tk:<35}: {d['acc']:.1%}  (n={d['n']})  {bar}")
    print(sep, flush=True)


def _print_comparison(entries: list[tuple[str, dict]]) -> None:
    all_mods = sorted({mod for _, m in entries for mod in m.get("per_modality", {})})
    lw, cw = 20, 12
    width  = lw + cw * (1 + len(all_mods)) + 2
    sep    = "=" * width
    header = f"  {'Label':<{lw}}" + f"{'Overall':>{cw}}" + \
             "".join(f"{mod:>{cw}}" for mod in all_mods)
    print()
    print(sep)
    print("  Held-out Comparison")
    print(sep)
    print(header)
    print("-" * width)
    for label, m in entries:
        row = f"  {label:<{lw}}{m['accuracy']*100:>{cw-1}.1f}%"
        for mod in all_mods:
            d = m.get("per_modality", {}).get(mod)
            row += f"{d['acc']*100:>{cw-1}.1f}%" if d else f"{'—':>{cw}}"
        print(row)
    print(sep, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-path",            required=True)
    ap.add_argument("--video-root",          required=True)
    ap.add_argument("--variant",             default="b_topk_nofusion")
    ap.add_argument("--ckpt-path",           default=None,
                    help="Checkpoint to load. Omit for untrained baseline.")
    ap.add_argument("--label",               required=True,
                    help="Run label (used for output subdir and comparison table).")
    ap.add_argument("--out-dir",             default="runs/heldout_eval")
    ap.add_argument("--num-samples",         type=int, default=150)
    ap.add_argument("--seed",                type=int, default=99,
                    help="Eval sampling seed (different from train seed).")
    ap.add_argument("--train-seed",          type=int, default=42)
    ap.add_argument("--train-max-samples",   type=int, default=2000)
    ap.add_argument("--every",               type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, idxs = _build_heldout(
        args.ann_path, args.video_root,
        args.train_max_samples, args.train_seed,
        args.num_samples, args.seed,
    )

    model   = _load_model(args.variant, args.ckpt_path)
    results = _run(model, ds, idxs, every=args.every)
    metrics = _compute(results)

    _print_summary(metrics, args.label)

    metrics_path = out_dir / "metrics.json"
    results_path = out_dir / "results.jsonl"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {metrics_path}", flush=True)

    # Auto-compare with any sibling runs
    siblings = []
    for d in sorted(Path(args.out_dir).iterdir()):
        mp = d / "metrics.json"
        if d.is_dir() and mp.exists() and d.name != args.label:
            with open(mp) as f:
                siblings.append((d.name, json.load(f)))
    if siblings:
        _print_comparison(siblings + [(args.label, metrics)])


if __name__ == "__main__":
    main()
