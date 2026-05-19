"""AVQA dataset: real AV clips + yes/no questions, matches the dummy contract.

Reads:
  - JSON in AVHBench schema: [{video_id, task, text, label}, ...]
  - mp4 clips at <video_root>/{youtube_id}.mp4
    where youtube_id = video_id with trailing "_NNNNNN" clip suffix stripped

Per item returns:
  videos:     (T=8, 3, 224, 224) fp16, CLIP-normalized
  audio_mels: (8, 1, 128, 204)   fp32, from ImageBind's load_and_transform_audio_data
  prompt:     str
  answer:     str ("Yes." or "No.")

Matches DummyAVDataset's output keys so the same train loop and collator work.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Make AVHBench code importable (same setup as encoders.py)
_AVHBENCH_ROOT = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVHBench" / "AVHBench-Align-FT"
if str(_AVHBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_AVHBENCH_ROOT))

from video_llama.processors.video_processor import load_video  # noqa: E402
from video_llama.processors import transforms_video  # noqa: E402
from video_llama.models.ImageBind.data import load_and_transform_audio_data  # noqa: E402


# CLIP normalization stats — matches AlproVideoBaseProcessor in AVHBench
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def _strip_clip_suffix(video_id: str) -> str:
    """'-HG3Omg_89c_000030' -> '-HG3Omg_89c'"""
    return re.sub(r"_\d{6}$", "", video_id)


class AVQADataset(Dataset):
    """AVQA yes/no dataset.

    Args:
        ann_path: JSON in AVHBench schema (list of {video_id, task, text, label}).
        video_root: directory containing {youtube_id}.mp4 files.
        n_frames: number of frames to sample per clip (default 8).
        img_size: spatial resolution (default 224).
        normalize_label: whether to append '.' to labels (matches dummy contract).
    """

    NUM_FRAMES: int = 8
    IMG_SIZE: int = 224

    def __init__(
        self,
        ann_path: str | Path,
        video_root: str | Path,
        n_frames: int = NUM_FRAMES,
        img_size: int = IMG_SIZE,
        normalize_label: bool = True,
    ):
        self.ann_path = Path(ann_path)
        self.video_root = Path(video_root)
        self.n_frames = n_frames
        self.img_size = img_size
        self.normalize_label = normalize_label

        with open(self.ann_path) as f:
            self.records = json.load(f)

        # Video transforms: resize -> center crop -> uint8 -> /255 -> normalize.
        # Mirrors AVHBench's eval-time processor. Input: (C, T, H, W) float.
        self.video_transform = transforms.Compose([
            transforms_video.NormalizeVideo(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def _load_video_tensor(self, video_path: Path) -> torch.Tensor:
        """Decord-load + transform a clip. Returns (T, 3, 224, 224) fp16."""
        # load_video returns (C, T, H, W) float at original resolution
        frms = load_video(
            video_path=str(video_path),
            n_frms=self.n_frames,
            height=self.img_size,
            width=self.img_size,
            sampling="uniform",
        )
        # Normalize expects float in [0,1] in (C, T, H, W).
        frms = frms / 255.0
        frms = self.video_transform(frms)               # (C, T, H, W)
        frms = frms.permute(1, 0, 2, 3).contiguous()    # (T, C, H, W)
        return frms.to(torch.float16)

    def _load_audio_mel(self, video_path: Path) -> torch.Tensor:
        """ImageBind audio pipeline -> (N_clips=8, 1, 128, 204) fp32 on CPU."""
        # ImageBind expects a list and returns (B, N_clips, 1, 128, 204).
        # Use CPU device so the dataset works inside dataloader workers.
        mel = load_and_transform_audio_data([str(video_path)], device=torch.device("cpu"))
        return mel.squeeze(0).contiguous()  # (N_clips, 1, 128, 204)

    def __getitem__(self, idx: int) -> dict:
        # Robust: on any video/audio decode error, skip to next index
        # rather than crashing the dataloader worker. Caps at 10 retries.
        for offset in range(10):
            j = (idx + offset) % len(self.records)
            rec = self.records[j]
            if "video_root" in rec:
                video_path = Path(rec["video_root"]) / f"{rec['video_id']}.mp4"
            else:
                yt_id = _strip_clip_suffix(rec["video_id"])
                video_path = self.video_root / f"{yt_id}.mp4"
            if not video_path.exists():
                continue
            try:
                videos = self._load_video_tensor(video_path)
                audio_mels = self._load_audio_mel(video_path)
            except Exception as e:
                # decord / ImageBind / PyAV errors all caught here
                if offset == 0:
                    print(f"[AVQADataset] skipping idx={j} ({yt_id}): "
                          f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            label = rec["label"]
            if self.normalize_label and not label.endswith("."):
                label = label + "."
            return {
                "videos": videos,
                "audio_mels": audio_mels,
                "prompt": rec["text"],
                "answer": label,
            }
        raise RuntimeError(f"AVQADataset: 10 consecutive bad items starting at idx={idx}")
