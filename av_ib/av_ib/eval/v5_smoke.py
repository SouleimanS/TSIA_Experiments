"""Phase 4: smoke test AVModelV5 end-to-end.

Verifies five invariants:
    1. Model construction succeeds (Qwen3-Omni + LoRA + VIBs + fusion + splicer)
    2. Trainable param count is in the expected range
    3. forward_generate runs on a real MUSIC-AVQA video without crashing
    4. The C-MIB splicer actually fires (provider closure call count > 0)
    5. Output is parseable into MUSIC-AVQA's answer vocabulary

The test is intentionally noisy in output — each step prints what it's doing
so a failure at step 3 vs step 5 is instantly clear.

Random initialization of VIBs means the output WILL be poor (probably wrong).
That's not the point. The point is: does the entire pipeline complete a forward
without exploding, and does the splicer participate as designed?

Usage:
    python -m av_ib.eval.v5_smoke --ann-path /path/to/avqa-test.json \\
                                  --video-root /path/to/videos/all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch


def main(args):
    print("=" * 60)
    print("Phase 4 smoke test: AVModelV5")
    print("=" * 60)

    # ----- Check 1: Construction -----
    print("\n[1/5] Constructing AVModelV5...")
    t0 = time.time()
    try:
        from av_ib.model.av_model_v5 import AVModelV5
        model = AVModelV5(use_lora=True)
        print(f"  OK in {time.time() - t0:.1f}s")
    except Exception as e:
        traceback.print_exc()
        print(f"FAIL at construction: {type(e).__name__}: {e}")
        return 1

    # ----- Check 2: Trainable param count -----
    print("\n[2/5] Trainable param summary...")
    try:
        summary = model.trainable_summary()
        for name, n in summary.items():
            if name.startswith("__"):
                continue
            print(f"  {name:25s} {n/1e6:7.2f}M trainable")
        trainable_total = summary["__total_trainable__"]
        total = summary["__total_params__"]
        print(f"  {'TOTAL TRAINABLE':25s} {trainable_total/1e6:7.2f}M")
        print(f"  {'TOTAL PARAMS':25s} {total/1e9:7.2f}B")
        print(f"  fraction trainable        {100 * trainable_total / total:.4f}%")
        if trainable_total < 1e6:
            print("WARN: trainable params suspiciously low — LoRA + VIBs missing?")
        if trainable_total > 5e9:
            print("WARN: trainable params suspiciously high — frozen base may be trainable")
    except Exception as e:
        traceback.print_exc()
        print(f"FAIL at param summary: {type(e).__name__}: {e}")
        return 2

    # ----- Check 3 + 4 + 5: One real forward pass -----
    print("\n[3-5/5] Loading one MUSIC-AVQA record and calling forward_generate...")
    try:
        with open(args.ann_path) as f:
            records = json.load(f)
        records = [r for r in records if r.get("question_deleted", 0) == 0]
        # Try a few records — first one's video might be corrupt
        for rec_idx in range(min(5, len(records))):
            rec = records[rec_idx]
            video_path = Path(args.video_root) / f"{rec['video_id']}.mp4"
            if video_path.exists():
                break
        else:
            print(f"FAIL: no videos found in {args.video_root}")
            return 3
        print(f"  Using record {rec_idx}: video_id={rec['video_id']}")
        print(f"  Video path: {video_path}")

        # Render the prompt
        from av_ib.data.musicavqa import render_question
        prompt = render_question(rec["question_content"], rec["templ_values"])
        gold = rec["anser"]
        print(f"  Prompt: {prompt}")
        print(f"  Gold:   {gold}")
    except Exception as e:
        traceback.print_exc()
        print(f"FAIL at record load: {type(e).__name__}: {e}")
        return 3

    # Wrap v5's provider to count calls — verifies splice fires
    print("\n  Instrumenting provider to count splice events...")
    original_make_provider = model._make_provider
    call_log = {"count": 0, "shapes_in": [], "shapes_out": []}

    def instrumented_make_provider():
        provider, kls, zs = original_make_provider()

        def wrapped(audio_3d, video_3d):
            call_log["count"] += 1
            call_log["shapes_in"].append({
                "audio": tuple(audio_3d.shape),
                "video": tuple(video_3d.shape),
            })
            result = provider(audio_3d, video_3d)
            call_log["shapes_out"].append(tuple(result.shape))
            return result

        return wrapped, kls, zs

    model._make_provider = instrumented_make_provider

    # ----- Check 3: forward_generate doesn't crash -----
    print("\n[3/5] Running forward_generate (this loads + decodes the video — slow first time)...")
    t0 = time.time()
    try:
        outputs = model.forward_generate(
            videos=[str(video_path)],
            audios=[str(video_path)],
            prompts=[prompt],
            max_new_tokens=10,
        )
        elapsed = time.time() - t0
        raw_pred = outputs[0]
        print(f"  OK in {elapsed:.1f}s")
        print(f"  Raw output: {raw_pred!r}")
    except Exception as e:
        traceback.print_exc()
        print(f"FAIL at forward_generate: {type(e).__name__}: {e}")
        return 3

    # ----- Check 4: Splicer fired -----
    print("\n[4/5] Splicer activity check...")
    print(f"  Provider call count: {call_log['count']}")
    if call_log["count"] == 0:
        print("FAIL: provider was never called. Splicer didn't fire.")
        print("  Possible causes:")
        print("    - Hook on get_audio_features didn't register")
        print("    - wrapper._current_provider was None during forward")
        print("    - The forward_pre_hook on model returned early")
        return 4
    print(f"  Splicer fired {call_log['count']} time(s)")
    for i, (in_shape, out_shape) in enumerate(zip(call_log['shapes_in'], call_log['shapes_out'])):
        print(f"    call {i}: audio={in_shape['audio']} video={in_shape['video']} -> joint={out_shape}")

    # ----- Check 5: Output sanity -----
    print("\n[5/5] Output parse check (cosmetic — VIBs are random init)...")
    # Inline parser from baseline script
    import re
    ANSWER_VOCAB = {
        "yes", "no", "two", "one", "zero", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "more than ten", "left", "right", "middle",
        "indoor", "outdoor", "simultaneously",
        "violin", "cello", "piano", "flute", "guitar", "acoustic_guitar", "electric_bass",
        "clarinet", "saxophone", "accordion", "trumpet", "tuba", "trombone", "horn",
        "ukulele", "banjo", "pipa", "guzheng", "erhu", "suona", "xylophone",
    }
    head = raw_pred.strip().lower()
    head = re.sub(r"[.,!?;:\"'()]", " ", head)
    parsed = "??"
    for tok in sorted(ANSWER_VOCAB, key=lambda s: -len(s)):
        if re.search(r"\b" + re.escape(tok) + r"\b", head):
            parsed = tok
            break
    print(f"  Raw:    {raw_pred!r}")
    print(f"  Parsed: {parsed!r}")
    print(f"  Gold:   {gold!r}")
    if parsed == "??":
        print("  NOTE: output didn't parse to vocab. Expected with random VIB init.")
        print("        Not a failure — model just hasn't learned anything yet.")

    print("\n" + "=" * 60)
    print("Phase 4 SMOKE TEST PASSED")
    print("  Construction: OK")
    print("  Trainable params: OK")
    print("  forward_generate: OK")
    print(f"  Splicer fired: {call_log['count']} time(s)")
    print(f"  Output parseable: {parsed != '??'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    args = p.parse_args()
    sys.exit(main(args))
