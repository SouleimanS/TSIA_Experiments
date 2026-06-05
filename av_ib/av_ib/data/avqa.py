"""AVQA (original multi-choice) dataset for Qwen3-Omni.

Annotation format (avqa-train.json):
    [
      {
        "video_id": "00000001",
        "question_id": 1,
        "question_content": "What is the sound made by the instrument?",
        "anser": "guitar",          # note: typo in original annotation is intentional
        ...
      },
      ...
    ]

The question is framed as an open-ended prompt; the model must produce the
answer string. For multi-choice format the prompt includes the options
(if available) so the model can reference them.

Returns paths, not tensors — Qwen3-Omni's processor does its own loading.
"""
from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class AVQADataset(Dataset):
    """Original AVQA multi-choice dataset.

    Args:
        ann_path:    path to avqa-train.json (or test/val)
        video_root:  directory containing <video_id>.mp4 files
        skip_missing: skip records whose video file is absent (default True)
    """

    def __init__(self, ann_path, video_root, skip_missing: bool = True):
        self.video_root = Path(video_root)
        with open(ann_path) as f:
            records = json.load(f)

        if skip_missing:
            before = len(records)
            records = [r for r in records
                       if (self.video_root / f"{r['video_id']}.mp4").exists()]
            dropped = before - len(records)
            if dropped:
                print(f"  [AVQADataset] dropped {dropped} records with missing videos.")

        self.records = records
        print(f"  [AVQADataset] {len(self.records)} records from {ann_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        for offset in range(10):
            j = (idx + offset) % len(self.records)
            rec = self.records[j]
            video_path = self.video_root / f"{rec['video_id']}.mp4"
            if not video_path.exists():
                continue
            prompt = self._build_prompt(rec)
            return {
                "video_path": str(video_path),
                "audio_path": str(video_path),
                "prompt": prompt,
                "answer": rec["anser"],
                "meta": {
                    "video_id": rec["video_id"],
                    "question_id": rec.get("question_id"),
                },
            }
        raise RuntimeError(f"AVQADataset: 10 consecutive missing videos at idx={idx}")

    @staticmethod
    def _build_prompt(rec: dict) -> str:
        q = rec["question_content"].strip()
        # Include choices if present in the annotation
        if "options" in rec and rec["options"]:
            opts = "  ".join(f"({chr(65+i)}) {o}" for i, o in enumerate(rec["options"]))
            return f"{q}\nOptions: {opts}"
        return q
