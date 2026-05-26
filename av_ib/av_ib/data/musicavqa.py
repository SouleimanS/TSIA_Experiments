"""MUSIC-AVQA dataset for Qwen3-Omni.

Returns raw file paths; Qwen3-Omni's processor handles mel-spec and frame sampling.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from torch.utils.data import Dataset


_PLACEHOLDER_RE = re.compile(r"<[A-Za-z][A-Za-z0-9_]*>")


def render_question(question_content: str, templ_values_str: str) -> str:
    try:
        values = json.loads(templ_values_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"templ_values not valid JSON: {templ_values_str!r}") from e
    if not isinstance(values, list):
        raise ValueError(f"templ_values must be a list, got {values!r}")
    placeholders = _PLACEHOLDER_RE.findall(question_content)
    if len(placeholders) != len(values):
        raise ValueError(
            f"Placeholder count mismatch: question has {len(placeholders)}, "
            f"templ_values has {len(values)}. Question: {question_content!r}"
        )
    out = question_content
    for v in values:
        out = _PLACEHOLDER_RE.sub(str(v), out, count=1)
    return out


def parse_type(type_str: str) -> tuple[str, str]:
    parts = json.loads(type_str)
    if not isinstance(parts, list) or len(parts) != 2:
        raise ValueError(f"type must be 2-element list, got {parts!r}")
    return parts[0], parts[1]


class MusicAVQADataset(Dataset):
    def __init__(self, ann_path, video_root, skip_deleted: bool = True):
        self.ann_path = Path(ann_path)
        self.video_root = Path(video_root)
        with open(self.ann_path) as f:
            records = json.load(f)
        if skip_deleted:
            records = [r for r in records if r.get("question_deleted", 0) == 0]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        for offset in range(10):
            j = (idx + offset) % len(self.records)
            rec = self.records[j]
            video_path = self.video_root / f"{rec['video_id']}.mp4"
            if not video_path.exists():
                continue
            try:
                prompt = render_question(rec["question_content"], rec["templ_values"])
            except Exception as e:
                if offset == 0:
                    print(f"[MusicAVQADataset] skipping idx={j}: {e}", flush=True)
                continue
            return {
                "video_path": str(video_path),
                "audio_path": str(video_path),
                "prompt": prompt,
                "answer": rec["anser"],
                "meta": {
                    "video_id": rec["video_id"],
                    "question_id": rec["question_id"],
                    "type": rec["type"],
                },
            }
        raise RuntimeError(f"MusicAVQADataset: 10 bad items starting at idx={idx}")
