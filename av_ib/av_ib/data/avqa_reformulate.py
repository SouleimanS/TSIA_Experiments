"""Reformulate AVQA multi-choice records as AVHBench-style Yes/No examples.

Each AVQA record produces 2 output records (1 Yes + 1 No), so the final
dataset is 1:1 balanced. The output JSON format matches AVHBench's
qa.json exactly, so the same eval/data loader can read both.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List


def reformulate_one(record: dict, rng: random.Random) -> List[dict]:
    """One AVQA record -> 2 AVHBench-style Yes/No records."""
    question = record["question_text"]
    options = record["multi_choice"]
    correct_idx = record["answer"]
    video_id = record["video_name"]

    correct_answer = options[correct_idx]
    distractors = [opt for i, opt in enumerate(options) if i != correct_idx]
    chosen_distractor = rng.choice(distractors)

    yes_record = {
        "video_id": video_id,
        "task": "AVQA_Yes_No",
        "text": question + " Is the answer '" + correct_answer + "'?",
        "label": "Yes",
    }
    no_record = {
        "video_id": video_id,
        "task": "AVQA_Yes_No",
        "text": question + " Is the answer '" + chosen_distractor + "'?",
        "label": "No",
    }
    return [yes_record, no_record]


def reformulate_all(records: List[dict], seed: int = 0) -> List[dict]:
    """List of AVQA records -> list of AVHBench-style records."""
    rng = random.Random(seed)
    out = []
    for rec in records:
        out.extend(reformulate_one(rec, rng))
    return out


def load_and_reformulate(
    avqa_json_path,
    output_json_path,
    seed: int = 0,
) -> int:
    """Load AVQA JSON, reformulate, write to disk. Returns output count."""
    with open(avqa_json_path) as f:
        records = json.load(f)
    out = reformulate_all(records, seed=seed)
    with open(output_json_path, "w") as f:
        json.dump(out, f, indent=2)
    return len(out)
