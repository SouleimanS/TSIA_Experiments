"""Inference for AVHBench using AVHModel-Align-FT.

Adapted from inference.py (which is the AffectGPT emotion-reasoning loop).
The model-loading and chat machinery are reused unchanged; only the data
loop is replaced to read AVHBench's qa.json (video_id / task / text / label)
and emit one CSV row per QA record.

Usage:
    python inference_avhbench.py \
        --cfg-path eval_configs/video_llama_eval_withaudio_stage3.yaml \
        --ckpt models/AVHBench/checkpoint_000002_loss_0.291.pth \
        --video-root /path/to/AVHBench_v0/video \
        --qa-json /path/to/AVHBench_v0/json/qa.json \
        --output predictions.csv \
        --gpu-id 0

Notes:
    * The cfg-path file sets up Vicuna + BLIP-2 + ImageBind + the two
      Video-LLaMA branches. The --ckpt flag adds the AVHBench fine-tune
      on top (cfg.model_cfg.ckpt_3 — their convention).
    * One record at a time, fresh chat_state per record. The model has
      no cross-record memory; we don't want one question contaminating
      the next.
    * Resumes from existing output CSV if it exists, so re-running after
      a crash doesn't redo completed records.
"""
import os
import json
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import decord
decord.bridge.set_bridge('torch')

from video_llama.tasks import *
from video_llama.models import *
from video_llama.runners import *
from video_llama.processors import *
from video_llama.datasets.builders import *
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import (
    Chat, Conversation, default_conversation, SeparatorStyle,
)


def setup_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def upload_video_fresh(chat, video_path: str):
    """Open a fresh conversation and upload a video. Returns (chat_state, img_list).

    Each AVHBench record gets its own conversation so prior records'
    context cannot leak in. Subtitle is always None for AVHBench."""
    chat_state = Conversation(
        system="",
        roles=("Human", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="###",
    )
    img_list = []
    chat.upload_video(video_path, chat_state, img_list, subtitle=None)
    return chat_state, img_list


def load_done_ids(output_csv: Path) -> set[tuple[str, str, str]]:
    """Return a set of (video_id, task, text) tuples already processed.

    Used to skip records that have predictions in a prior partial run.
    Key is (video_id, task, text) because the same video appears with
    multiple questions across multiple tasks.
    """
    if not output_csv.exists():
        return set()
    done = set()
    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["video_id"], row["task"], row["text"]))
    return done


def main():
    parser = argparse.ArgumentParser(description="AVHBench inference with AVHModel-Align-FT")
    parser.add_argument("--cfg-path", required=True,
                        help="Path to eval YAML (e.g. eval_configs/video_llama_eval_withaudio_stage3.yaml)")
    parser.add_argument("--ckpt", required=True,
                        help="Path to AVHBench fine-tune checkpoint (.pth). Set as cfg.model_cfg.ckpt_3.")
    parser.add_argument("--video-root", required=True,
                        help="Directory containing {video_id}.mp4 files")
    parser.add_argument("--qa-json", required=True,
                        help="Path to AVHBench qa.json")
    parser.add_argument("--output", required=True,
                        help="Output CSV path. Resumes if file exists.")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None,
                        help="Optional: only process the first N records (for smoke testing)")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--options", nargs="+",
                        help="Override config settings: --options key=value")
    args = parser.parse_args()

    # The Config class expects args.cfg_path attribute name (dash -> underscore).
    args.cfg_path = args.cfg_path  # already correct from argparse
    cfg = Config(args)

    # Slot in the AVHBench fine-tune as ckpt_3, following their convention.
    cfg.model_cfg.ckpt_3 = args.ckpt
    cfg.model_cfg.device_8bit = args.gpu_id

    setup_seeds(42)

    print("=" * 70)
    print("Loading model...")
    print(f"  config:      {args.cfg_path}")
    print(f"  AVHBench FT: {args.ckpt}")
    print(f"  device:      cuda:{args.gpu_id}")
    print("=" * 70)

    model_cls = registry.get_model_class(cfg.model_cfg.arch)
    model = model_cls.from_config(cfg.model_cfg).to(f"cuda:{args.gpu_id}")
    model = model.eval()

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device=f"cuda:{args.gpu_id}")

    # Load QA records
    qa_path = Path(args.qa_json)
    with open(qa_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {qa_path}")

    if args.max_records is not None:
        records = records[:args.max_records]
        print(f"Limited to {len(records)} records via --max-records")

    # Resume support: skip records already in the output CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(output_path)
    if done:
        print(f"Resuming: {len(done)} records already in {output_path}")

    video_root = Path(args.video_root)

    # Open CSV in append mode so resume works without rewriting prior rows.
    file_exists = output_path.exists()
    fout = open(output_path, "a", newline="")
    writer = csv.DictWriter(
        fout,
        fieldnames=["video_id", "task", "text", "gold_label", "prediction"],
    )
    if not file_exists:
        writer.writeheader()
        fout.flush()

    n_done = 0
    n_skipped = 0
    n_errors = 0

    try:
        for rec in tqdm(records, desc="Inference"):
            video_id = str(rec["video_id"])
            task = rec["task"]
            text = rec["text"]
            gold = rec["label"]

            key = (video_id, task, text)
            if key in done:
                n_skipped += 1
                continue

            video_path = video_root / f"{video_id}.mp4"
            if not video_path.exists():
                # Don't crash the whole run for one missing file; log and continue.
                print(f"[WARN] Missing video: {video_path}")
                writer.writerow({
                    "video_id": video_id, "task": task, "text": text,
                    "gold_label": gold, "prediction": "[ERROR: missing video]",
                })
                fout.flush()
                n_errors += 1
                continue

            try:
                chat_state, img_list = upload_video_fresh(chat, str(video_path))
                chat.ask(text, chat_state)
                response, _ = chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    max_length=2000,
                )
            except Exception as e:
                print(f"[ERROR] {video_id} / {task[:30]}... : {type(e).__name__}: {e}")
                response = f"[ERROR: {type(e).__name__}]"
                n_errors += 1

            writer.writerow({
                "video_id": video_id, "task": task, "text": text,
                "gold_label": gold, "prediction": response,
            })
            fout.flush()
            n_done += 1
    finally:
        fout.close()

    print("=" * 70)
    print(f"Done.   processed: {n_done}   skipped(resume): {n_skipped}   errors: {n_errors}")
    print(f"Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
