"""Diagnostic eval: AV Matching task only, with rotating prompt paraphrases.

Hypothesis: tonight's sweep collapsed on AVHBench AV Matching (acc ~0.5, all
outputs a single token) despite the model emitting balanced Yes/No on AVQA-val.
The AVHBench AV Matching prompt is a single fixed string for all 1876 items,
so the model has no text-side variation to ground against. This script tests
whether varying the prompt surface form breaks the collapse.

Usage:
    python scripts/eval_avhbench_paraphrase.py \
        --variant v3 --ckpt-dir runs/avqa_v3_3ep_b1e-2 --out-name v3_b1e-2

Output: results/avhbench_paraphrase_{out_name}.{json,csv}
  - per-paraphrase accuracy, Yes%, and confusion counts
  - per-item predictions with which paraphrase was used

Notes:
  - Paraphrase index 5 is *negated* ("Is there a mismatch..."), so for those
    items the expected label is flipped. The script handles this and reports
    label-aligned accuracy throughout.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from av_ib.eval.avhbench import _load_video, _load_audio, _build_prompt


HOME = Path.home()
AVHBENCH_ROOT = HOME / "SOULEIMAN_repo" / "datasets" / "AVHBench" / "data" / "AVHBench_v0"
AVHBENCH_QA = AVHBENCH_ROOT / "json" / "qa.json"
AVHBENCH_VIDEOS = AVHBENCH_ROOT / "video"
RESULTS_DIR = Path("results")


# Paraphrases. Index 0 is the original. Index 5 is negated — label flips.
PARAPHRASES = [
    # 0: original
    {"text": "Are the contexts of audio and visual content matching?",
     "negated": False},
    # 1: simpler, common phrasing
    {"text": "Does the audio match what is happening in the video?",
     "negated": False},
    # 2: rephrased as alignment
    {"text": "Is what you hear consistent with what you see?",
     "negated": False},
    # 3: emphasizes correspondence
    {"text": "Do the sounds in the audio align with the visual content?",
     "negated": False},
    # 4: emphasizes shared source
    {"text": "Considering the audio and video together, are they from the same source?",
     "negated": False},
    # 5: NEGATED — Yes means mismatch, so label flips
    {"text": "Is there a mismatch between the audio and the video?",
     "negated": True},
]


def parse_yes_no(text: str) -> str:
    import re
    head = text.strip().lower()[:30]
    if re.search(r"\byes\b", head):
        return "Yes"
    if re.search(r"\bno\b", head):
        return "No"
    return "??"


def flip(label: str) -> str:
    return "No" if label == "Yes" else "Yes" if label == "No" else label


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
    """Compute acc / Yes% / confusion on label-aligned predictions.

    `records` here use `effective_label` (label flipped for negated paraphrase).
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    tp = sum(1 for r in records if r["effective_label"] == "Yes" and r["pred"] == "Yes")
    tn = sum(1 for r in records if r["effective_label"] == "No"  and r["pred"] == "No")
    fp = sum(1 for r in records if r["effective_label"] == "No"  and r["pred"] == "Yes")
    fn = sum(1 for r in records if r["effective_label"] == "Yes" and r["pred"] == "No")
    qq = sum(1 for r in records if r["pred"] == "??")
    acc = (tp + tn) / n
    yes_pct = (tp + fp) / n * 100
    return {
        "n": n, "acc": acc, "yes_pct": yes_pct,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "non_yes_no": qq,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["v1", "v2", "v3"])
    ap.add_argument("--ckpt-dir", type=str, required=True,
                    help="Checkpoint dir, e.g. runs/avqa_v3_3ep_b1e-2")
    ap.add_argument("--out-name", type=str, required=True,
                    help="Output basename suffix, e.g. v3_b1e-2")
    ap.add_argument("--max-items", type=int, default=None,
                    help="Cap items (for smoke testing)")
    ap.add_argument("--test-file", type=str, default=None,
                    help="Override test JSON path (default: AVHBench full qa.json)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    qa_path = Path(args.test_file) if args.test_file else AVHBENCH_QA
    print(f"Loading items from {qa_path}", flush=True)
    with open(qa_path) as f:
        all_items = json.load(f)

    # AV Matching only, with Yes/No labels
    items = [d for d in all_items
             if d.get("task") == "AV Matching" and d.get("label") in ("Yes", "No")]
    print(f"AV Matching items: {len(items)} (filtered from {len(all_items)} total)", flush=True)

    if args.max_items:
        items = items[: args.max_items]
        print(f"  Smoke test: limiting to {len(items)} items", flush=True)

    print(f"\nBuilding model: {args.variant}", flush=True)
    model = build_model(args.variant, device)
    ckpt = Path(args.ckpt_dir) / "best.pt"
    load_checkpoint(model, ckpt, device)
    model.eval()

    print(f"\nRunning paraphrase eval on {len(items)} AV Matching items "
          f"with {len(PARAPHRASES)} rotating paraphrases...", flush=True)
    print(f"  paraphrase index i = item_idx % {len(PARAPHRASES)}", flush=True)
    for i, p in enumerate(PARAPHRASES):
        tag = " (NEGATED)" if p["negated"] else ""
        print(f"  [{i}]{tag}  {p['text']}", flush=True)
    print()

    predictions = []
    n_failed = 0
    t0 = time.time()
    for i, rec in enumerate(items):
        vid_path = AVHBENCH_VIDEOS / f"{rec['video_id']}.mp4"
        p_idx = i % len(PARAPHRASES)
        para = PARAPHRASES[p_idx]
        effective_label = flip(rec["label"]) if para["negated"] else rec["label"]
        try:
            videos = _load_video(str(vid_path), device)
            audios = _load_audio(str(vid_path), device)
            prompt = _build_prompt("AV Matching", para["text"])
            with torch.no_grad():
                out = model.forward_generate(videos, audios, [prompt], max_new_tokens=10)[0]
            pred = parse_yes_no(out)
            predictions.append({
                "video_id": rec["video_id"],
                "paraphrase_idx": p_idx,
                "paraphrase_text": para["text"],
                "negated": para["negated"],
                "orig_label": rec["label"],
                "effective_label": effective_label,
                "pred": pred,
                "raw": out[:80],
            })
        except Exception as e:
            n_failed += 1
            predictions.append({
                "video_id": rec["video_id"],
                "paraphrase_idx": p_idx,
                "paraphrase_text": para["text"],
                "negated": para["negated"],
                "orig_label": rec["label"],
                "effective_label": effective_label,
                "pred": "FAIL",
                "raw": str(e)[:80],
            })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(items) - i - 1) / rate
            n_correct = sum(1 for p in predictions if p["pred"] == p["effective_label"])
            print(f"  [{i+1}/{len(items)}]  acc-so-far={n_correct/(i+1):.4f}  "
                  f"failed={n_failed}  {rate:.1f} it/s  eta={eta/60:.1f}min",
                  flush=True)

    total_time = time.time() - t0
    print(f"\nEval done in {total_time/60:.1f} min  failed={n_failed}", flush=True)

    # Per-paraphrase breakdown
    by_para = defaultdict(list)
    for p in predictions:
        if p["pred"] != "FAIL":
            by_para[p["paraphrase_idx"]].append(p)

    results = {
        "variant": args.variant,
        "ckpt_dir": str(args.ckpt_dir),
        "n_items": len(items),
        "n_failed": n_failed,
        "eval_time_s": total_time,
        "paraphrases": [
            {"idx": i, "text": p["text"], "negated": p["negated"]}
            for i, p in enumerate(PARAPHRASES)
        ],
        "per_paraphrase": {},
        "overall": {},
    }

    print("\n=== Per-paraphrase metrics (label-aligned) ===")
    for p_idx in sorted(by_para):
        recs = by_para[p_idx]
        m = compute_metrics(recs)
        results["per_paraphrase"][str(p_idx)] = {
            "text": PARAPHRASES[p_idx]["text"],
            "negated": PARAPHRASES[p_idx]["negated"],
            **m,
        }
        neg_tag = " [NEG]" if PARAPHRASES[p_idx]["negated"] else "     "
        print(f"  [{p_idx}]{neg_tag} n={m['n']:4d}  acc={m['acc']:.4f}  "
              f"Yes%={m['yes_pct']:5.1f}  "
              f"TP={m['TP']:3d} TN={m['TN']:3d} FP={m['FP']:3d} FN={m['FN']:3d}  "
              f"?? {m['non_yes_no']}")
        print(f"        {PARAPHRASES[p_idx]['text']!r}")

    # Overall (label-aligned)
    all_recs = [p for p in predictions if p["pred"] != "FAIL"]
    m = compute_metrics(all_recs)
    results["overall"] = m
    print(f"\n  OVERALL:  n={m['n']}  acc={m['acc']:.4f}  Yes%={m['yes_pct']:.1f}")

    out_json = RESULTS_DIR / f"avhbench_paraphrase_{args.out_name}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {out_json}", flush=True)

    out_csv = RESULTS_DIR / f"avhbench_paraphrase_{args.out_name}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video_id", "paraphrase_idx", "paraphrase_text", "negated",
            "orig_label", "effective_label", "pred", "raw",
        ])
        w.writeheader()
        w.writerows(predictions)
    print(f"Saved predictions to {out_csv}", flush=True)


if __name__ == "__main__":
    main()