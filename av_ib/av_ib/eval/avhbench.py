"""AVHBench evaluation harness for av_ib models.

Reuses AVHBench's preprocessing code (frame sampling, audio mel-spec) so
that what we feed our model matches the test conditions of the
published baseline as closely as possible.

Pipeline per record:
    1. Load qa.json record: {video_id, task, text, label}
    2. Decode 8 video frames from MP4 (using the AVHBench `alpro_video_eval` processor).
    3. Compute audio mel-spec (8 clips x 1 x 128 x 204) using AVHBench's ImageBind data loader.
    4. Build prompt for the task.
    5. Call model.forward_generate -> generated text.
    6. Save (video_id, task, text, label, prediction) to CSV.

Public API:
    run_eval(model, qa_json_path, video_dir, output_csv, max_records=None,
             max_new_tokens=50, batch_size=1)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn


# Reuse AVHBench's preprocessing
_AVHBENCH_ROOT = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVHBench" / "AVHBench-Align-FT"
if str(_AVHBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_AVHBENCH_ROOT))

from video_llama.processors.video_processor import AlproVideoEvalProcessor  # noqa: E402
from video_llama.models.ImageBind import data as imagebind_data  # noqa: E402


# Build the video processor once; it's stateless.
_VIDEO_PROCESSOR = AlproVideoEvalProcessor(image_size=224, n_frms=8)


def _build_prompt(task: str, text: str) -> str:
    """Map task to the AVHBench prompt template.

    For v1 we use the same minimal Vicuna-style template as in llm.py.
    The text field already contains the question for judgment tasks, and
    'Describe what you see and hear.' for captioning.
    """
    return text


def _load_video(path: str, device: torch.device) -> torch.Tensor:
    """Decode 8 frames from MP4, return tensor (1, 8, 3, 224, 224) fp16."""
    # AlproVideoEvalProcessor returns shape (C, T, H, W); we permute to (T, C, H, W)
    # and add batch dim.
    video = _VIDEO_PROCESSOR(path)            # (3, 8, 224, 224)
    video = video.permute(1, 0, 2, 3)         # (8, 3, 224, 224)
    return video.unsqueeze(0).to(device, dtype=torch.float16)


def _load_audio(path: str, device: torch.device) -> torch.Tensor:
    """Compute audio mel-spec, return tensor (1, 8, 1, 128, 204) fp32."""
    # imagebind_data.load_and_transform_audio_data does the whole pipeline:
    # PyAV decode -> resample -> mel-spec (8 clips).
    audio = imagebind_data.load_and_transform_audio_data([path], device)
    # Shape: (1, 8, 1, 128, 204).
    return audio


def run_eval(
    model: nn.Module,
    qa_json_path: str | Path,
    video_dir: str | Path,
    output_csv: str | Path,
    max_records: Optional[int] = None,
    max_new_tokens: int = 50,
    device: str = "cuda",
    print_every: int = 10,
) -> str:
    """Run inference on AVHBench records, write predictions to CSV.

    CSV columns: idx, video_id, task, text, label, prediction
    Returns the output CSV path.
    """
    qa_json_path = Path(qa_json_path)
    video_dir = Path(video_dir)
    output_csv = Path(output_csv)

    with open(qa_json_path) as f:
        records = json.load(f)
    if max_records is not None:
        records = records[:max_records]

    print(f"Loaded {len(records)} records from {qa_json_path}")
    print(f"Video dir: {video_dir}")
    print(f"Output:    {output_csv}\n")

    model.eval()
    out_f = open(output_csv, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(["idx", "video_id", "task", "text", "label", "prediction"])

    t0 = time.time()
    n_failed = 0
    for i, rec in enumerate(records):
        video_id = rec["video_id"]
        task = rec["task"]
        text = rec["text"]
        label = rec["label"]
        video_path = video_dir / f"{video_id}.mp4"

        try:
            videos = _load_video(str(video_path), device)
            audio_mels = _load_audio(str(video_path), device)
            prompt = _build_prompt(task, text)
            preds = model.forward_generate(
                videos, audio_mels, [prompt], max_new_tokens=max_new_tokens
            )
            prediction = preds[0]
        except Exception as e:
            prediction = f"<ERROR: {type(e).__name__}: {str(e)[:100]}>"
            n_failed += 1

        writer.writerow([i, video_id, task, text, label, prediction])
        out_f.flush()

        if (i + 1) % print_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate
            print(f"  [{i+1}/{len(records)}] {rate:.2f} ex/s  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  failed={n_failed}")

    out_f.close()
    total = time.time() - t0
    print(f"\nDone in {total:.0f}s ({len(records)/total:.2f} ex/s)  failed={n_failed}")
    return str(output_csv)
