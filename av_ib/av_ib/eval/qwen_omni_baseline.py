"""Vanilla Qwen3-Omni baseline on MUSIC-AVQA val.

No VIB, no LoRA, no training. Just runs out-of-the-box Qwen3-Omni inference
to establish a ceiling/floor reference for the C-MIB experiments.

Uses the existing answer-vocab parser from musicavqa_eval.py so the numbers
are directly comparable to the Vicuna-era v1-v4 results.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional
from qwen_omni_utils import process_mm_info
import torch

from av_ib.data.musicavqa import render_question, parse_type

# Inlined from musicavqa_eval to avoid pulling in AVHBench's video_llama import chain
# (which is broken under transformers>=5.x). The functions themselves are pure-Python.
import re
ANSWER_VOCAB = {
    "yes", "no", "two", "one", "zero", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "more than ten",
    "left", "right", "middle",
    "indoor", "outdoor",
    "simultaneously",
    "violin", "cello", "piano", "flute", "guitar", "acoustic_guitar", "electric_bass",
    "clarinet", "saxophone", "accordion", "trumpet", "tuba", "trombone", "horn", "ukulele",
    "banjo", "pipa", "guzheng", "erhu", "suona", "xylophone",
}


DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}


def parse_answer(text: str, vocab: set = ANSWER_VOCAB) -> str:
    head = text.strip().lower()
    head = re.sub(r"[.,!?;:\"'()]", " ", head)
    # Normalize standalone digits to spelled-out words BEFORE vocab matching
    head = re.sub(r"\b(\d+)\b", lambda m: DIGIT_TO_WORD.get(m.group(1), m.group(1)), head)
    for tok in sorted(vocab, key=lambda s: -len(s)):
        candidates = [tok]
        if "_" in tok:
            candidates.append(tok.replace("_", " "))
        for c in candidates:
            if re.search(r"\b" + re.escape(c) + r"\b", head):
                return tok
    return "??"


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


def build_conversation(video_path: str, prompt: str) -> list:
    """Single-turn AV question following Qwen3-Omni's expected schema."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt + " Answer with one word from the allowed vocabulary."},
            ],
        },
    ]


def run_baseline(
    ann_path: str | Path,
    video_root: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    max_records: Optional[int] = None,
    max_new_tokens: int = 10,
    fps: int = 1,
    print_every: int = 10,
) -> dict:
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    ann_path = Path(ann_path)
    video_root = Path(video_root)
    output_csv = Path(output_csv)
    output_json = Path(output_json)

    print(f"Loading {model_path} ...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Skip speech generation — we only want text answers
    if hasattr(model, "disable_talker"):
        model.disable_talker()
        print("  Talker disabled (text-only inference).")
    model.eval()

    with open(ann_path) as f:
        records = json.load(f)
    records = [r for r in records if r.get("question_deleted", 0) == 0]
    if max_records is not None:
        records = records[:max_records]
    print(f"Evaluating on {len(records)} records.")

    out_f = open(output_csv, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(["idx", "video_id", "type", "prompt", "gold", "raw_pred", "parsed_pred", "correct"])

    n_correct = n_total = n_failed = n_missing = non_match = 0
    by_modality: dict[str, list[int]] = {}
    t0 = time.time()

    for i, rec in enumerate(records):
        video_path = video_root / f"{rec['video_id']}.mp4"
        if not video_path.exists():
            n_missing += 1
            continue
        try:
            prompt = render_question(rec["question_content"], rec["templ_values"])
            conv = build_conversation(str(video_path), prompt)
            # Two-step: render text with placeholders, then extract media,
            # then combine in processor() — gives correct audio_lengths.
            text = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False,
            )
            audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            ).to(model.device).to(model.dtype)
            with torch.no_grad():
                text_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_audio=False,
                    use_audio_in_video=True,
                )
            # Strip the prompt prefix from the generated ids
            input_len = inputs["input_ids"].shape[1]
            gen_ids = text_ids[:, input_len:] if text_ids.dim() == 2 else text_ids[input_len:]
            raw_pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            parsed = parse_answer(raw_pred, ANSWER_VOCAB)
            gold = rec["anser"]
            correct = int(parsed == gold)
            if parsed == "??":
                non_match += 1

            n_correct += correct
            n_total += 1
            try:
                modality, _ = parse_type(rec["type"])
            except ValueError:
                modality = "?"
            by_modality.setdefault(modality, [0, 0])
            by_modality[modality][1] += 1
            by_modality[modality][0] += correct

            writer.writerow([i, rec["video_id"], rec["type"], prompt, gold, raw_pred, parsed, correct])
            out_f.flush()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            n_failed += 1
            # Print first failure's full traceback to log for diagnosis
            if n_failed == 1:
                print(f"\n=== FIRST FAILURE TRACEBACK (idx={i}, video_id={rec['video_id']}) ===")
                print(tb)
                print("=== END TRACEBACK ===\n", flush=True)
            err_msg = f"<ERROR: {type(e).__name__}: {str(e)[:200]} | tb: {tb.splitlines()[-3] if len(tb.splitlines()) >= 3 else 'n/a'}>"
            writer.writerow([i, rec["video_id"], rec.get("type", "?"), "", "",
                             err_msg, "??", 0])
            out_f.flush()

        if (i + 1) % print_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate if rate > 0 else 0
            acc = n_correct / max(n_total, 1)
            print(f"  [{i+1}/{len(records)}] acc={acc:.3f}  "
                  f"rate={rate:.2f} ex/s  eta={eta:.0f}s  failed={n_failed} missing={n_missing}")

    out_f.close()

    summary = {
        "model": model_path,
        "ann_path": str(ann_path),
        "n_records": len(records),
        "n_total": n_total,
        "n_correct": n_correct,
        "n_failed": n_failed,
        "n_missing_video": n_missing,
        "non_match": non_match,
        "accuracy": n_correct / max(n_total, 1),
        "per_modality": {
            m: {"n": tot, "n_correct": corr, "acc": corr / tot if tot else 0.0}
            for m, (corr, tot) in sorted(by_modality.items())
        },
        "elapsed_s": time.time() - t0,
    }
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFinal: acc={summary['accuracy']:.4f}  ({n_correct}/{n_total})")
    print(f"Per-modality: {summary['per_modality']}")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--output-csv", default="qwen_omni_baseline.csv")
    p.add_argument("--output-json", default="qwen_omni_baseline.json")
    p.add_argument("--model-path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--max-records", type=int, default=None,
                   help="Cap for smoke testing; omit for full val")
    p.add_argument("--max-new-tokens", type=int, default=10)
    p.add_argument("--fps", type=int, default=1)
    args = p.parse_args()

    run_baseline(
        ann_path=args.ann_path,
        video_root=args.video_root,
        output_csv=args.output_csv,
        output_json=args.output_json,
        model_path=args.model_path,
        max_records=args.max_records,
        max_new_tokens=args.max_new_tokens,
        fps=args.fps,
    )
