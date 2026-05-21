"""Periodic validation eval against MUSIC-AVQA val split, called from training loops.

Top-1 accuracy against the closed 41-token MUSIC-AVQA vocab, plus per-modality
breakdown (Audio / Visual / Audio-Visual). Used by all three trainers
(v1, v2, v3) on MUSIC-AVQA so they share scoring logic.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import torch

from av_ib.data.musicavqa import render_question, parse_type
from av_ib.eval.avhbench import _load_video, _load_audio, _build_prompt


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


def parse_answer(text: str, vocab: set[str] = ANSWER_VOCAB) -> str:
    head = text.strip().lower()
    head = re.sub(r"[.,!?;:\"'()]", " ", head)
    for tok in sorted(vocab, key=lambda s: -len(s)):
        candidates = [tok]
        if "_" in tok:
            candidates.append(tok.replace("_", " "))
        for c in candidates:
            if re.search(r"\b" + re.escape(c) + r"\b", head):
                return tok
    return "??"


def musicavqa_val_eval(
    model,
    val_records: list,
    video_root: Path,
    device: str,
    step: int,
    *,
    max_records: Optional[int] = None,
    vocab: set[str] = ANSWER_VOCAB,
) -> dict:
    model.eval()
    if max_records is not None:
        val_records = val_records[:max_records]

    n_correct = n_total = n_failed = n_missing = non_match = 0
    by_modality = defaultdict(lambda: [0, 0])
    t0 = time.time()

    for rec in val_records:
        vid_path = video_root / f"{rec['video_id']}.mp4"
        if not vid_path.exists():
            n_missing += 1
            continue
        try:
            videos = _load_video(str(vid_path), device)
            audios = _load_audio(str(vid_path), device)
            prompt = render_question(rec["question_content"], rec["templ_values"])
            with torch.no_grad():
                pred = model.forward_generate(videos, audios, [prompt], max_new_tokens=10)[0]
            pred_label = parse_answer(pred, vocab)
            if pred_label == "??":
                non_match += 1
            if pred_label == rec["anser"]:
                n_correct += 1
            n_total += 1
            try:
                modality, _ = parse_type(rec["type"])
            except ValueError:
                modality = "?"
            by_modality[modality][1] += 1
            if pred_label == rec["anser"]:
                by_modality[modality][0] += 1
        except Exception:
            n_failed += 1

    model.train()
    per_modality = {
        m: {"n": tot, "n_correct": corr, "acc": (corr / tot if tot else 0.0)}
        for m, (corr, tot) in sorted(by_modality.items())
    }
    return {
        "step": step,
        "accuracy": n_correct / max(n_total, 1),
        "n_correct": n_correct,
        "n_total": n_total,
        "n_failed": n_failed,
        "n_missing_video": n_missing,
        "non_match": non_match,
        "eval_time_s": time.time() - t0,
        "per_modality": per_modality,
    }
