"""Dump labels + decoded tokens for one record to verify forward_train masking.

For each of 3 records, prints:
    - The full input_ids decoded
    - The prompt_len boundary
    - Which tokens are masked (-100) vs trained on
    - The decoded answer span
    - Any structural mismatch (e.g., answer starts at index != prompt_len)

If labels are correct, the "trained-on" tokens should EXACTLY match the answer text
(plus maybe an EOS). If anything else appears, masking is broken.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main(args):
    print("=" * 60)
    print("Label-masking inspection for forward_train")
    print("=" * 60)

    print("\n[1/2] Loading AVModelV5...")
    from av_ib.model.av_model_v5 import AVModelV5
    model = AVModelV5(use_lora=True)
    tokenizer = model.qwen.tokenizer

    print("\n[2/2] Loading records...")
    with open(args.ann_path) as f:
        records = json.load(f)
    records = [r for r in records if r.get("question_deleted", 0) == 0]
    video_root = Path(args.video_root)

    # Pick 3 records with different answer types
    picked = []
    seen_answers = set()
    for r in records:
        if (video_root / f"{r['video_id']}.mp4").exists() and r["anser"] not in seen_answers:
            picked.append(r)
            seen_answers.add(r["anser"])
            if len(picked) >= 3:
                break

    from av_ib.data.musicavqa import render_question
    for idx, rec in enumerate(picked):
        print("\n" + "=" * 60)
        print(f"RECORD {idx}: video_id={rec['video_id']}  gold={rec['anser']!r}")
        print("=" * 60)

        video_path = str(video_root / f"{rec['video_id']}.mp4")
        prompt = render_question(rec["question_content"], rec["templ_values"])
        answer = rec["anser"]

        # Use the wrapper's own _prep_inputs to ensure we test the actual path
        inputs, prompt_len = model.qwen._prep_inputs(video_path, prompt, answer=answer)
        input_ids = inputs["input_ids"]              # (1, L)
        L = input_ids.shape[1]

        # Reproduce the labels exactly as forward_train builds them
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        # Decode the full input as the model sees it
        full_decoded = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        print(f"\nFull input ({L} tokens), with prompt_len={prompt_len}:")
        print(f"  {full_decoded[:200]!r}...{full_decoded[-200:]!r}")

        # The masked (prompt) portion
        prompt_ids = input_ids[0, :prompt_len].tolist()
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        print(f"\nMasked-out prompt tokens (positions 0..{prompt_len-1}):")
        print(f"  Last 80 chars: ...{prompt_text[-80:]!r}")

        # The trained-on (label != -100) portion
        train_mask = labels[0] != -100
        train_ids = input_ids[0][train_mask].tolist()
        train_text = tokenizer.decode(train_ids, skip_special_tokens=False)
        train_text_clean = tokenizer.decode(train_ids, skip_special_tokens=True)
        print(f"\nTrained-on tokens (label != -100, count={train_mask.sum().item()}):")
        print(f"  ids:          {train_ids}")
        print(f"  decoded raw:  {train_text!r}")
        print(f"  decoded clean: {train_text_clean!r}")
        print(f"  expected (gold): {answer!r}")

        # Check the boundary
        boundary_window = input_ids[0, max(0, prompt_len-3):prompt_len+5].tolist()
        boundary_text = tokenizer.decode(boundary_window, skip_special_tokens=False)
        print(f"\nBoundary window (3 before, 5 after prompt_len={prompt_len}):")
        print(f"  ids:     {boundary_window}")
        print(f"  decoded: {boundary_text!r}")

        # Diagnose
        clean_match = train_text_clean.strip() == answer.strip()
        contains_extra = len(train_text_clean.strip()) > len(answer.strip()) + 5  # tolerate EOS-ish
        print(f"\nDiagnosis:")
        print(f"  trained text EQUALS gold answer (after strip)?  {clean_match}")
        print(f"  trained text contains substantially MORE than answer? {contains_extra}")
        if not clean_match and not contains_extra:
            print(f"  -> trained text differs slightly from gold (could be just whitespace/EOS)")
        if contains_extra:
            print(f"  -> ⚠️  trained on MORE than just the answer — masking too permissive")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    args = p.parse_args()
    sys.exit(main(args))
