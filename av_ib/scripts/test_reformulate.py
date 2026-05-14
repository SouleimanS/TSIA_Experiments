"""Test AVQA -> AVHBench reformulation with hand-crafted mock data."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from av_ib.data.avqa_reformulate import reformulate_all


MOCK_AVQA_RECORDS = [
    {
        "id": 1205,
        "video_name": "0AEJTlHIhz0_000358",
        "video_id": 1467,
        "question_text": "Why do the people in the video scream?",
        "multi_choice": ["Roller coaster", "On a pirate ship", "Take Ferris wheel", "Take the jumping machine"],
        "answer": 0,
        "question_relation": "Both",
        "question_type": "Why",
    },
    {
        "id": 1206,
        "video_name": "XYZabcDEF_000123",
        "video_id": 1468,
        "question_text": "What instrument is making the sound?",
        "multi_choice": ["Piano", "Violin", "Drums", "Guitar"],
        "answer": 2,
        "question_relation": "Sound",
        "question_type": "Which",
    },
    {
        "id": 1207,
        "video_name": "abc123_000999",
        "video_id": 1469,
        "question_text": "Where is the dog located?",
        "multi_choice": ["On the couch", "In the kitchen", "By the door", "Under the table"],
        "answer": 3,
        "question_relation": "View",
        "question_type": "Where",
    },
]


def main():
    print("=== Input: 3 AVQA records ===\n")
    for r in MOCK_AVQA_RECORDS:
        print(f"  Q: {r['question_text']}")
        print(f"  Options: {r['multi_choice']}")
        print(f"  Correct: {r['multi_choice'][r['answer']]}")
        print()

    out = reformulate_all(MOCK_AVQA_RECORDS, seed=42)
    print(f"\n=== Output: {len(out)} reformulated records ===\n")
    for r in out:
        print(f"  task={r['task']}, video_id={r['video_id']}")
        print(f"    text:  {r['text']}")
        print(f"    label: {r['label']}")
        print()

    # Sanity checks
    n_yes = sum(1 for r in out if r["label"] == "Yes")
    n_no = sum(1 for r in out if r["label"] == "No")
    n_total = len(out)
    print(f"\n=== Counts ===")
    print(f"  Total:  {n_total} (expected {2 * len(MOCK_AVQA_RECORDS)})")
    print(f"  Yes:    {n_yes} (expected {len(MOCK_AVQA_RECORDS)})")
    print(f"  No:     {n_no} (expected {len(MOCK_AVQA_RECORDS)})")
    assert n_total == 2 * len(MOCK_AVQA_RECORDS), "wrong total count"
    assert n_yes == len(MOCK_AVQA_RECORDS), "wrong yes count"
    assert n_no == len(MOCK_AVQA_RECORDS), "wrong no count"

    # Determinism check
    out2 = reformulate_all(MOCK_AVQA_RECORDS, seed=42)
    assert out == out2, "reformulation should be deterministic"

    out3 = reformulate_all(MOCK_AVQA_RECORDS, seed=99)
    assert out != out3, "different seed should produce different distractors"
    print("\n  Determinism: PASS (same seed -> same output, diff seed -> diff output)")

    print("\nAll reformulation tests passed.")


if __name__ == "__main__":
    main()
