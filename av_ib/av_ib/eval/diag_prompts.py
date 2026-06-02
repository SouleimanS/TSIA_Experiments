#!/usr/bin/env python3
"""Diagnostic prompts — targeted stress tests for AVModelV6.

These are not random MUSIC-AVQA questions. Each probe isolates ONE hypothesized
failure mode so a wrong answer points to a specific mechanism, not generic error.

The probe taxonomy maps directly to the known failure signature from the n=1000
tier-1 results (b loses on yes/no & count, wins on spatial/relational), and to
the central thesis of the WVS paper your project builds on: video-capable omni
models hallucinate audio from visual priors (the "Clever Hans" effect).

Probe families
──────────────
 A. AUDIO-GROUNDING (does it hear, or guess from video?)
    Pair each video with a question only answerable from the audio stream.
    A model that scores well on the matched-audio version but flips its answer
    when audio is muted/swapped is using a visual shortcut.

 B. VISUAL-SHORTCUT TRAP
    Questions whose visually-obvious answer is wrong for the actual audio.
    Catches the Clever Hans failure directly.

 C. COUNTING under occlusion / off-screen sound
    "How many instruments are sounding?" where some are heard but not seen.
    Catches the count regression (b: 0.71 vs baseline 0.79).

 D. SPATIAL / RELATIONAL  (b's strength — confirm it holds)
    "Which side is the louder instrument on?" — needs A+V binding.

 E. EXISTENCE / NEGATION
    "Is there a piano sound?" with and without the piano audible.
    Yes/no calibration probe (b's weak spot).

 F. TEMPORAL ORDER
    "Which instrument starts first?" — needs temporal audio tracking.

Usage — generate a probe set as a MUSIC-AVQA-style json the eval can consume:
  python diag_prompts.py --emit-json results/diagnostics/probe_set.json

Usage — just print the prompts (to paste into any chat / eval):
  python diag_prompts.py --print
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Each probe: (family, modality, qclass, question, what_a_wrong_answer_means)
PROBES = [
    # ── A. Audio grounding ───────────────────────────────────────────────
    ("A-audio-grounding", "Audio", "Existential",
     "Listen to the audio only. Is there a string instrument being played? Answer yes or no.",
     "Wrong → not actually attending to the audio stream; guessing from frame content."),
    ("A-audio-grounding", "Audio", "Comparative",
     "Based only on what you hear, is the louder sound coming from a wind or a string instrument?",
     "Wrong → audio loudness comparison not grounded; defaulting to a visual prior."),

    # ── B. Visual-shortcut trap ──────────────────────────────────────────
    ("B-visual-shortcut", "Audio-Visual", "Existential",
     "An instrument is visible but may be silent. Is that visible instrument actually producing sound right now? Answer yes or no.",
     "Wrong (says yes for a silent-but-visible instrument) → classic Clever Hans visual shortcut."),
    ("B-visual-shortcut", "Audio", "Existential",
     "Ignore what instruments you can see. Judging purely by ear, is a drum being struck in this clip? Answer yes or no.",
     "Wrong → visual presence overriding the actual audio evidence."),

    # ── C. Counting ──────────────────────────────────────────────────────
    ("C-counting", "Audio", "Counting",
     "By sound alone, how many distinct instruments can you hear playing?",
     "Wrong → cannot separate concurrent audio sources; likely counts visible instruments instead."),
    ("C-counting", "Audio-Visual", "Counting",
     "How many instruments are producing sound, including any you can hear but not see?",
     "Wrong (matches only the visible count) → ignoring off-screen audio sources."),

    # ── D. Spatial / relational (b's strength) ───────────────────────────
    ("D-spatial", "Audio-Visual", "Location",
     "Which side of the frame is the instrument that is currently making sound on — left, right, or middle?",
     "Wrong → audio-visual spatial binding broken (this is where variant b normally excels)."),
    ("D-spatial", "Audio-Visual", "Comparative",
     "Of the two instruments visible, which one is the one you can actually hear?",
     "Wrong → cannot bind the heard sound to its visual source."),

    # ── E. Existence / negation (yes/no calibration) ─────────────────────
    ("E-negation", "Audio", "Existential",
     "Is it true that NO piano sound occurs anywhere in this clip? Answer yes or no.",
     "Wrong → negation handling / yes-bias (b over-predicts on y/n)."),
    ("E-negation", "Audio-Visual", "Existential",
     "Is there at least one moment where everything is silent? Answer yes or no.",
     "Wrong → silence detection; tests whether 'audio present' is a blind prior."),

    # ── F. Temporal order ────────────────────────────────────────────────
    ("F-temporal", "Audio", "Temporal",
     "Which instrument's sound begins first in the clip?",
     "Wrong → no temporal tracking of the audio stream; answering from a static visual guess."),
    ("F-temporal", "Audio-Visual", "Temporal",
     "Does the instrument on the left start playing before or after the one on the right?",
     "Wrong → temporal + spatial audio-visual binding failure."),
]


def emit_json(path: Path, video_id_placeholder: str = "REPLACE_VIDEO_ID") -> None:
    """Emit a MUSIC-AVQA-schema json. You attach real video_ids per probe family
    by editing the file, or by pairing with curated clips. Left as a template so
    you control which clips stress which probe."""
    records = []
    for i, (family, modality, qclass, q, meaning) in enumerate(PROBES):
        records.append({
            "question_id": 900000 + i,
            "video_id": video_id_placeholder,
            "type": json.dumps([modality, qclass]),
            "question_content": q,
            "templ_values": "[]",
            "anser": "",                # fill with gold per attached clip
            "question_deleted": 0,
            "_probe_family": family,
            "_diagnostic_meaning": meaning,
        })
    path.write_text(json.dumps(records, indent=2))
    print(f"[emit] {len(records)} probes → {path}")
    print("       Replace REPLACE_VIDEO_ID and fill 'anser' per attached clip,")
    print("       then run eval_musicavqa.py / v6_eval.py against this file.")


def print_probes() -> None:
    fam = None
    for family, modality, qclass, q, meaning in PROBES:
        if family != fam:
            fam = family
            print(f"\n{'='*78}\n{family}\n{'='*78}")
        print(f"\n[{modality} · {qclass}]")
        print(f"  Q: {q}")
        print(f"  ↳ {meaning}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--emit-json", type=Path, default=None)
    p.add_argument("--print", action="store_true")
    args = p.parse_args()
    if args.emit_json:
        emit_json(args.emit_json)
    if args.print or not args.emit_json:
        print_probes()


if __name__ == "__main__":
    main()
