#!/usr/bin/env python3
"""Download the AVQA dataset (Yang et al., ACM MM 2022).

1. Annotations: fetched from the official GitHub release (verified URLs).
2. Videos: 10-second VGGSound clips cut from YouTube via yt-dlp.
   video_name = <YouTubeID>_<6-digit start seconds>.

A fraction of YouTube videos is gone (private/deleted/region-locked); those
are logged to dead_clips.txt and skipped — AVQADataset(skip_missing=True)
handles the holes. The script is resumable: existing .mp4 files are skipped.

Run on a machine with internet access (login node), e.g.:
    pip install yt-dlp
    nohup python download_avqa.py --out ~/SOULEIMAN_repo/datasets/AVQA \
        --workers 8 > avqa_download.log 2>&1 &

Options --split train|val|all and --limit N allow a small pilot download
first (recommended: --limit 2000 to unblock the e1 gate + miner while the
full crawl continues).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ANN_URLS = {
    "train": "https://raw.githubusercontent.com/AlyssaYoung/AVQA/main/data/annotation/train_qa.json",
    "val":   "https://raw.githubusercontent.com/AlyssaYoung/AVQA/main/data/annotation/val_qa.json",
}


def fetch_annotations(out_dir: Path) -> dict:
    ann = {}
    for split, url in ANN_URLS.items():
        dst = out_dir / f"{split}_qa.json"
        if not dst.exists():
            print(f"downloading {split} annotations ...")
            urllib.request.urlretrieve(url, dst)
        ann[split] = json.loads(dst.read_text())
        print(f"  {split}: {len(ann[split])} QA pairs")
    return ann


def clip_args(video_name: str):
    """<YouTubeID>_<SSSSSS> -> (youtube_id, start_s, end_s)."""
    yid, start = video_name.rsplit("_", 1)
    s = int(start)
    return yid, s, s + 10


def download_clip(video_name: str, video_dir: Path) -> tuple[str, bool]:
    dst = video_dir / f"{video_name}.mp4"
    if dst.exists():
        return video_name, True
    yid, s, e = clip_args(video_name)
    cmd = [
        "yt-dlp",
        "--quiet", "--no-warnings",
        "-f", "bv*[height<=360]+ba/b[height<=360]/b",
        "--download-sections", f"*{s}-{e}",
        "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4",
        "-o", str(dst),
        f"https://www.youtube.com/watch?v={yid}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        ok = r.returncode == 0 and dst.exists()
    except subprocess.TimeoutExpired:
        ok = False
    if not ok and dst.exists():
        dst.unlink()  # no partial files
    return video_name, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dataset root directory")
    ap.add_argument("--split", choices=["train", "val", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="download at most N clips (0 = all); for pilots")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    video_dir = out / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    ann = fetch_annotations(out)
    splits = ["train", "val"] if args.split == "all" else [args.split]
    names = sorted({r["video_name"] for sp in splits for r in ann[sp]})
    todo = [n for n in names if not (video_dir / f"{n}.mp4").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(names)} unique clips; {len(todo)} to download")

    dead_log = out / "dead_clips.txt"
    n_ok = n_dead = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
            open(dead_log, "a") as df:
        futs = {ex.submit(download_clip, n, video_dir): n for n in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            name, ok = fut.result()
            if ok:
                n_ok += 1
            else:
                n_dead += 1
                df.write(name + "\n")
                df.flush()
            if i % 50 == 0:
                print(f"  [{i}/{len(todo)}] ok={n_ok} dead={n_dead}", flush=True)

    print(f"done: ok={n_ok} dead={n_dead} (dead clips -> {dead_log})")


if __name__ == "__main__":
    main()
