"""Evaluate v1/v2/v3 checkpoints on MUSIC-AVQA (native schema).

Loads checkpoint, runs forward_generate on the MUSIC-AVQA val or test split,
parses the answer against MUSIC-AVQA's closed answer vocab (41 tokens),
and reports overall + per-modality + per-reasoning-type accuracy.

Per-category breakdown is the key payload — see the 9 (modality, reasoning) buckets
in the JSON output.

Usage:
    python scripts/eval_musicavqa.py --variant v3 --ckpt-dir runs/avqa_v3_3ep_b1e-2 \
        --out-name v3_b1e-2 --split val
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from av_ib.data.musicavqa import render_question, parse_type
from av_ib.eval.avhbench import _load_video, _load_audio, _build_prompt


HOME = Path.home()
MUSICAVQA_ROOT = HOME / "SOULEIMAN_repo" / "datasets" / "MUSIC-AVQA" / "MUSIC-AVQA" / "data" / "json_update"
MUSICAVQA_VIDEO_ROOT = HOME / "SOULEIMAN_repo" / "datasets" / "MUSIC-AVQA" / "videos" / "all"
RESULTS_DIR = Path("results")


# Pre-computed from json_update/avqa-val.json + avqa-test.json (both have 41 tokens).
# Kept here as a fallback; the script also re-derives it from the loaded split.
DEFAULT_ANSWER_VOCAB = {
    "yes", "no", "two", "one", "zero", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "more than ten",
    "left", "right", "middle",
    "indoor", "outdoor",
    "simultaneously",
    "violin", "cello", "piano", "flute", "guitar", "acoustic_guitar", "electric_bass",
    "clarinet", "saxophone", "accordion", "trumpet", "tuba", "trombone", "horn", "ukulele",
    "banjo", "pipa", "guzheng", "erhu", "suona", "xylophone",
}


def parse_answer(text: str, vocab: set[str]) -> str:
    """Match the first vocab token that appears in the model output.

    Strategy: lowercase, strip punctuation, then scan vocab tokens by descending
    length so multi-word answers ('more than ten', 'acoustic_guitar') match
    before their shorter substrings. Returns '??' if nothing matches.
    """
    import re
    head = text.strip().lower()
    # Replace common punctuation with spaces so word boundaries work
    head = re.sub(r"[.,!?;:\"'()]", " ", head)
    # Tokens with underscores need preserving for 'acoustic_guitar' style answers
    # so we DON'T strip underscores. Also try with spaces instead of underscores.

    # Sort vocab by length desc to match longest first
    for tok in sorted(vocab, key=lambda s: -len(s)):
        # Underscored vocab tokens: try matching both underscored and space-separated
        candidates = [tok]
        if "_" in tok:
            candidates.append(tok.replace("_", " "))
        for c in candidates:
            if re.search(r"\b" + re.escape(c) + r"\b", head):
                return tok
    return "??"


def build_model(variant: str, device: str):
    if variant == "v1":
        from av_ib.model.av_model_v1 import AVModelV1
        return AVModelV1(use_lora=True).to(device)
    if variant == "v2":
        from av_ib.model.av_model_v2 import AVModelV2
        return AVModelV2(use_lora=True).to(device)
    if variant == "v3":
        from av_ib.model.av_model_v3 import AVModelV3
        return AVModelV3(use_lora=True).to(device)
    raise ValueError(f"Unknown variant: {variant}")


def load_checkpoint(model, ckpt_path, device):
    print(f"Loading {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location=device)
    own = dict(model.named_parameters())
    n_loaded = n_missing = 0
    for k, v in sd.items():
        if k in own:
            own[k].data.copy_(v.data)
            n_loaded += 1
        else:
            n_missing += 1
    print(f"  Loaded {n_loaded}/{len(sd)} params  (missing: {n_missing})", flush=True)


def compute_metrics(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "acc": 0.0, "non_match": 0}
    correct = sum(1 for r in records if r["pred"] == r["label"])
    non_match = sum(1 for r in records if r["pred"] == "??")
    return {
        "n": n,
        "acc": correct / n,
        "n_correct": correct,
        "non_match": non_match,
        "pred_dist": dict(Counter(r["pred"] for r in records).most_common(5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["v1", "v2", "v3"])
    ap.add_argument("--ckpt-dir", type=str, required=True,
                    help="Checkpoint dir, e.g. runs/avqa_v3_3ep_b1e-2")
    ap.add_argument("--out-name", type=str, required=True,
                    help="Output basename suffix, e.g. v3_b1e-2")
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="MUSIC-AVQA split to evaluate on (default val).")
    ap.add_argument("--max-items", type=int, default=None,
                    help="Cap items (for smoke testing)")
    ap.add_argument("--ann-file", type=str, default=None,
                    help="Override annotation path (default: json_update/avqa-{split}.json)")
    ap.add_argument("--video-root", type=str, default=str(MUSICAVQA_VIDEO_ROOT),
                    help="Video root (default: datasets/MUSIC-AVQA/videos/raw)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    ann_path = Path(args.ann_file) if args.ann_file else MUSICAVQA_ROOT / f"avqa-{args.split}.json"
    print(f"Loading items from {ann_path}", flush=True)
    with open(ann_path) as f:
        items = json.load(f)
    items = [r for r in items if r.get("question_deleted", 0) == 0]
    print(f"  {len(items)} items (after dropping question_deleted=1)", flush=True)

    if args.max_items:
        items = items[: args.max_items]
        print(f"  Smoke test: limiting to {len(items)} items", flush=True)

    # Derive answer vocab from the loaded split (more accurate than the default)
    vocab = set(r["anser"] for r in items) | DEFAULT_ANSWER_VOCAB
    print(f"  Answer vocab size: {len(vocab)}", flush=True)

    # Type distribution diagnostic
    type_counts = Counter(r["type"] for r in items)
    print(f"  Type distribution (top 10): {type_counts.most_common(10)}", flush=True)

    print(f"\nBuilding model: {args.variant}", flush=True)
    model = build_model(args.variant, device)
    ckpt = Path(args.ckpt_dir) / "best.pt"
    load_checkpoint(model, ckpt, device)
    model.eval()

    video_root = Path(args.video_root)
    print(f"\nVideo root: {video_root}", flush=True)
    print(f"Running eval on {len(items)} items...", flush=True)

    predictions = []
    n_failed = n_missing_video = 0
    t0 = time.time()

    for i, rec in enumerate(items):
        vid_path = video_root / f"{rec['video_id']}.mp4"
        if not vid_path.exists():
            n_missing_video += 1
            predictions.append({
                "video_id": rec["video_id"],
                "question_id": rec["question_id"],
                "type": rec["type"],
                "prompt": rec["question_content"],
                "label": rec["anser"],
                "pred": "MISSING_VIDEO",
                "raw": "",
            })
            continue
        try:
            prompt = render_question(rec["question_content"], rec["templ_values"])
            videos = _load_video(str(vid_path), device)
            audios = _load_audio(str(vid_path), device)
            with torch.no_grad():
                out = model.forward_generate(videos, audios, [prompt], max_new_tokens=10)[0]
            pred = parse_answer(out, vocab)
            predictions.append({
                "video_id": rec["video_id"],
                "question_id": rec["question_id"],
                "type": rec["type"],
                "prompt": prompt,
                "label": rec["anser"],
                "pred": pred,
                "raw": out[:80],
            })
        except Exception as e:
            n_failed += 1
            predictions.append({
                "video_id": rec["video_id"],
                "question_id": rec["question_id"],
                "type": rec["type"],
                "prompt": rec["question_content"],
                "label": rec["anser"],
                "pred": "FAIL",
                "raw": str(e)[:80],
            })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(items) - i - 1) / rate
            scored = [p for p in predictions if p["pred"] not in ("FAIL", "MISSING_VIDEO")]
            acc = sum(1 for p in scored if p["pred"] == p["label"]) / max(len(scored), 1)
            print(f"  [{i+1}/{len(items)}]  acc-so-far={acc:.4f}  "
                  f"failed={n_failed}  missing_vid={n_missing_video}  "
                  f"{rate:.1f} it/s  eta={eta/60:.1f}min", flush=True)

    total_time = time.time() - t0
    print(f"\nEval done in {total_time/60:.1f} min  "
          f"failed={n_failed}  missing_vid={n_missing_video}", flush=True)

    # Filter to scorable items only for metric computation
    scorable = [p for p in predictions if p["pred"] not in ("FAIL", "MISSING_VIDEO")]

    # Aggregate by (modality, reasoning) — 9 buckets in the full set
    by_full_type = defaultdict(list)
    by_modality = defaultdict(list)
    by_reasoning = defaultdict(list)
    for p in scorable:
        try:
            modality, reasoning = parse_type(p["type"])
        except ValueError:
            continue
        by_full_type[(modality, reasoning)].append(p)
        by_modality[modality].append(p)
        by_reasoning[reasoning].append(p)

    results = {
        "variant": args.variant,
        "ckpt_dir": str(args.ckpt_dir),
        "split": args.split,
        "n_items": len(items),
        "n_failed": n_failed,
        "n_missing_video": n_missing_video,
        "eval_time_s": total_time,
        "overall": compute_metrics(scorable),
        "per_modality": {},
        "per_reasoning": {},
        "per_type": {},
    }

    print("\n=== Overall ===")
    o = results["overall"]
    print(f"  n={o['n']}  acc={o['acc']:.4f}  non_match(??)={o['non_match']}")

    print("\n=== Per modality ===")
    for mod in sorted(by_modality):
        m = compute_metrics(by_modality[mod])
        results["per_modality"][mod] = m
        print(f"  {mod:14s}  n={m['n']:4d}  acc={m['acc']:.4f}  ??={m['non_match']}")

    print("\n=== Per reasoning type ===")
    for r in sorted(by_reasoning):
        m = compute_metrics(by_reasoning[r])
        results["per_reasoning"][r] = m
        print(f"  {r:14s}  n={m['n']:4d}  acc={m['acc']:.4f}  ??={m['non_match']}")

    print("\n=== Per (modality, reasoning) ===")
    for (mod, reas) in sorted(by_full_type):
        m = compute_metrics(by_full_type[(mod, reas)])
        results["per_type"][f"{mod}__{reas}"] = m
        print(f"  {mod:14s} x {reas:14s}  n={m['n']:4d}  acc={m['acc']:.4f}  ??={m['non_match']}")

    out_json = RESULTS_DIR / f"musicavqa_{args.out_name}_{args.split}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {out_json}", flush=True)

    out_csv = RESULTS_DIR / f"musicavqa_{args.out_name}_{args.split}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video_id", "question_id", "type", "prompt", "label", "pred", "raw",
        ])
        w.writeheader()
        w.writerows(predictions)
    print(f"Saved predictions to {out_csv}", flush=True)


if __name__ == "__main__":
    main()