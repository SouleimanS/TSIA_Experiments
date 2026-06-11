"""AVHBench QA dataset (VGGSound/AudioCaps clips) for Qwen3-Omni.

Annotation format (qa_final_single_filtered.json):
    [
      {
        "video_id": "01162",
        "task": "AV Matching",
        "text": "Are the contexts of audio and visual content matching?",
        "label": "No"
      },
      ...
    ]

Tasks are binary (Yes/No): A->V / V->A matching, audio-driven video
hallucination, video-driven audio hallucination. The clips are open-domain
(VGGSound + AudioCaps sources), so unlike MUSIC-AVQA the questions genuinely
require listening AND looking — the right substrate for the abstention
experiment.

Train/eval contamination guard: `split` partitions by video_id hash so no clip
appears in both. Returns paths, not tensors — Qwen3-Omni's processor does its
own loading.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

from torch.utils.data import Dataset


def _split_of(video_id: str, train_frac: float = 0.8) -> str:
    """Deterministic video-level split (stable across runs/machines)."""
    h = zlib.crc32(str(video_id).encode()) % 100
    return "train" if h < int(train_frac * 100) else "eval"


class AVHBenchQADataset(Dataset):
    """AVHBench binary QA over local VGGSound/AudioCaps clips.

    Args:
        ann_path:    path to qa_final_single_filtered.json
        video_root:  directory containing <video_id>.mp4 files
        split:       "train" / "eval" / "all" — video-level partition
        tasks:       optional list of task names to keep (default: all)
        skip_missing: drop records whose video file is absent (default True)
    """

    def __init__(self, ann_path, video_root, split: str = "all",
                 tasks=None, skip_missing: bool = True):
        self.video_root = Path(video_root)
        with open(ann_path) as f:
            records = json.load(f)

        if tasks:
            keep = set(tasks)
            records = [r for r in records if r.get("task") in keep]

        if split in ("train", "eval"):
            records = [r for r in records if _split_of(r["video_id"]) == split]

        if skip_missing:
            before = len(records)
            records = [r for r in records
                       if (self.video_root / f"{r['video_id']}.mp4").exists()]
            dropped = before - len(records)
            if dropped:
                print(f"  [AVHBenchQADataset] dropped {dropped} records "
                      f"with missing videos.")

        # synthetic stable per-question id: <video_id>#<index within video>
        seen: dict = {}
        for r in records:
            n = seen.get(r["video_id"], 0)
            r["question_id"] = f"{r['video_id']}#{n}"
            seen[r["video_id"]] = n + 1

        self.records = records
        by_task: dict = {}
        for r in records:
            by_task[r.get("task", "?")] = by_task.get(r.get("task", "?"), 0) + 1
        print(f"  [AVHBenchQADataset] {len(records)} records "
              f"(split={split}) from {ann_path}  tasks={by_task}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        video_path = self.video_root / f"{rec['video_id']}.mp4"
        prompt = f"{rec['text'].strip()} Answer yes or no."
        return {
            "video_path": str(video_path),
            "audio_path": str(video_path),
            "prompt": prompt,
            "answer": rec["label"],
            "meta": {
                "video_id": rec["video_id"],
                "question_id": rec["question_id"],
                "task": rec.get("task"),
            },
        }
