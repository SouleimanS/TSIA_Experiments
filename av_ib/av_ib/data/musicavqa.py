"""MUSIC-AVQA dataset: native schema (question_content + templ_values, free-form answer).

Reads:
  - JSON in MUSIC-AVQA schema (json_update/ variant):
    [{video_id, question_id, type, question_content, templ_values, question_deleted, anser}, ...]
  - mp4 clips at <video_root>/{video_id}.mp4
    (video_id is 8-digit zero-padded; prefix varies by split — vv, sa, eva, evv, va, esa for
    synthetic, or 8-digit plain for real — but file naming on disk matches the id exactly)

Per item returns (same dict keys as AVQADataset so av_collate works unchanged):
  videos:     (T=8, 3, 224, 224) fp16, CLIP-normalized
  audio_mels: (8, 1, 128, 204)   fp32, from ImageBind
  prompt:     str   — rendered question text (placeholders filled)
  answer:     str   — gold answer, lowercase, no trailing period (matches native vocab)

Differences from AVQADataset:
  - Native schema: no AVHBench-style reformulation, no distractor sampling. The model
    learns to emit one of MUSIC-AVQA's 41 answer tokens directly.
  - `type` field is exposed as `meta` for downstream per-category eval (not used in training).
  - Answer is NOT capitalized or '.'-terminated — MUSIC-AVQA's vocab is lowercase 'yes' / 'no'
    / 'two' / etc.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms

_AVHBENCH_ROOT = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVHBench" / "AVHBench-Align-FT"
if str(_AVHBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_AVHBENCH_ROOT))

from video_llama.processors.video_processor import load_video  # noqa: E402
from video_llama.processors import transforms_video  # noqa: E402
from video_llama.models.ImageBind.data import load_and_transform_audio_data  # noqa: E402


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# Placeholder pattern: matches <X>, <LR>, <LRer>, <Object>, <FB>, <TH>, etc.
# Anything between angle brackets containing letters / digits.
_PLACEHOLDER_RE = re.compile(r"<[A-Za-z][A-Za-z0-9_]*>")


def render_question(question_content: str, templ_values_str: str) -> str:
    """Fill MUSIC-AVQA template placeholders with their template values.

    Placeholders are <X>, <LR>, <Object>, etc. They are consumed left-to-right
    and replaced by successive entries from `templ_values`.

    Args:
        question_content: e.g. "Is the instrument on the <LR> louder than the instrument on the <LR>?"
        templ_values_str: e.g. '["left", "right"]' (JSON-encoded list of strings)

    Returns:
        Rendered string with all placeholders substituted.

    Raises:
        ValueError: if the number of placeholders != len(templ_values).
    """
    try:
        values = json.loads(templ_values_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"templ_values not valid JSON: {templ_values_str!r}") from e
    if not isinstance(values, list):
        raise ValueError(f"templ_values must be a list, got {type(values).__name__}: {values!r}")
    placeholders = _PLACEHOLDER_RE.findall(question_content)
    if len(placeholders) != len(values):
        raise ValueError(
            f"Placeholder count mismatch: question has {len(placeholders)} placeholders "
            f"{placeholders}, templ_values has {len(values)} entries {values}. "
            f"Question: {question_content!r}"
        )
    out = question_content
    for v in values:
        # str(v) is defensive — MUSIC-AVQA values are always strings but cheap insurance
        out = _PLACEHOLDER_RE.sub(str(v), out, count=1)
    return out


def parse_type(type_str: str) -> tuple[str, str]:
    """'["Audio-Visual", "Counting"]' -> ("Audio-Visual", "Counting")."""
    try:
        parts = json.loads(type_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"type not valid JSON: {type_str!r}") from e
    if not isinstance(parts, list) or len(parts) != 2:
        raise ValueError(f"type must be a 2-element list, got {parts!r}")
    return parts[0], parts[1]


class MusicAVQADataset(Dataset):
    """MUSIC-AVQA dataset with native schema.

    Args:
        ann_path: JSON in MUSIC-AVQA schema (json_update/avqa-*.json recommended).
        video_root: directory containing {video_id}.mp4 files.
        n_frames: number of frames to sample per clip (default 8).
        img_size: spatial resolution (default 224).
        skip_deleted: drop records with question_deleted==1 (default True).
    """

    NUM_FRAMES: int = 8
    IMG_SIZE: int = 224

    def __init__(
        self,
        ann_path: str | Path,
        video_root: str | Path,
        n_frames: int = NUM_FRAMES,
        img_size: int = IMG_SIZE,
        skip_deleted: bool = True,
    ):
        self.ann_path = Path(ann_path)
        self.video_root = Path(video_root)
        self.n_frames = n_frames
        self.img_size = img_size

        with open(self.ann_path) as f:
            records = json.load(f)
        if skip_deleted:
            records = [r for r in records if r.get("question_deleted", 0) == 0]
        self.records = records

        self.video_transform = transforms.Compose([
            transforms_video.NormalizeVideo(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def _load_video_tensor(self, video_path: Path) -> torch.Tensor:
        frms = load_video(
            video_path=str(video_path),
            n_frms=self.n_frames,
            height=self.img_size,
            width=self.img_size,
            sampling="uniform",
        )
        frms = frms / 255.0
        frms = self.video_transform(frms)               # (C, T, H, W)
        frms = frms.permute(1, 0, 2, 3).contiguous()    # (T, C, H, W)
        return frms.to(torch.float16)

    def _load_audio_mel(self, video_path: Path) -> torch.Tensor:
        mel = load_and_transform_audio_data([str(video_path)], device=torch.device("cpu"))
        return mel.squeeze(0).contiguous()  # (N_clips, 1, 128, 204)

    def __getitem__(self, idx: int) -> dict:
        for offset in range(10):
            j = (idx + offset) % len(self.records)
            rec = self.records[j]
            video_path = self.video_root / f"{rec['video_id']}.mp4"
            if not video_path.exists():
                continue
            try:
                videos = self._load_video_tensor(video_path)
                audio_mels = self._load_audio_mel(video_path)
                prompt = render_question(rec["question_content"], rec["templ_values"])
            except Exception as e:
                if offset == 0:
                    print(f"[MusicAVQADataset] skipping idx={j} ({rec['video_id']}): "
                          f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            answer = rec["anser"]  # native lowercase, no period
            return {
                "videos": videos,
                "audio_mels": audio_mels,
                "prompt": prompt,
                "answer": answer,
                # Extra metadata for per-category eval (collator drops or passes through)
                "meta": {
                    "video_id": rec["video_id"],
                    "question_id": rec["question_id"],
                    "type": rec["type"],
                },
            }
        raise RuntimeError(f"MusicAVQADataset: 10 consecutive bad items starting at idx={idx}")