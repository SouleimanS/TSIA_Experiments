"""AVQA (Yang et al., ACM MM 2022 — VGGSound-based) dataset for Qwen3-Omni.

Annotation format (train_qa.json / val_qa.json, verified against the official
GitHub release github.com/AlyssaYoung/AVQA):

    [
      {
        "id": 183,
        "video_name": "-HG3Omg_89c_000030",
        "video_id": 341,
        "question_text": "What happened in the video?",
        "multi_choice": ["motorboat", "Yacht consignment",
                          "Sailboat set sail", "Consignment car"],
        "answer": 1,                      # 0-3 index into multi_choice
        "question_relation": "View",      # Both | Sound | View
        "question_type": "Happening"
      },
      ...
    ]

video_name = <YouTubeID>_<6-digit start seconds>: a 10-second VGGSound clip.
Videos are expected as <video_root>/<video_name>.mp4.

The question is rendered as 4-way multiple choice with letter options; the
gold answer is the letter (A-D) so that exact-match scoring is robust to the
noisy option strings. Returns paths, not tensors — Qwen3-Omni's processor does
its own loading.
"""
from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

_LETTERS = ("A", "B", "C", "D")


def render_prompt(rec: dict) -> str:
    q = rec["question_text"].strip()
    opts = "  ".join(f"({_LETTERS[i]}) {o}"
                     for i, o in enumerate(rec["multi_choice"]))
    return (f"{q}\nOptions: {opts}\n"
            f"Answer with the letter of the correct option.")


class AVQADataset(Dataset):
    """AVQA multiple-choice QA over VGGSound clips.

    Args:
        ann_path:     train_qa.json or val_qa.json
        video_root:   directory containing <video_name>.mp4 files
        skip_missing: drop records whose clip is absent (default True;
                      a fraction of VGGSound clips is no longer downloadable)
    """

    def __init__(self, ann_path, video_root, skip_missing: bool = True):
        self.video_root = Path(video_root)
        with open(ann_path) as f:
            records = json.load(f)

        if skip_missing:
            before = len(records)
            records = [r for r in records
                       if (self.video_root / f"{r['video_name']}.mp4").exists()]
            dropped = before - len(records)
            if dropped:
                print(f"  [AVQADataset] dropped {dropped}/{before} records "
                      f"with missing videos.")

        # stable per-question id for the miner (records have a unique "id")
        for r in records:
            r["question_id"] = r["id"]

        self.records = records
        print(f"  [AVQADataset] {len(self.records)} records from {ann_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        video_path = self.video_root / f"{rec['video_name']}.mp4"
        gold_idx = int(rec["answer"])
        return {
            "video_path": str(video_path),
            "audio_path": str(video_path),
            "prompt": render_prompt(rec),
            "answer": _LETTERS[gold_idx],
            "meta": {
                "video_id": rec["video_name"],
                "question_id": rec["question_id"],
                "question_type": rec.get("question_type"),
                "question_relation": rec.get("question_relation"),
                "answer_text": rec["multi_choice"][gold_idx],
            },
        }
