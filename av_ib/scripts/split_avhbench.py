"""Split AVHBench yes/no QA items 80/20 by video for combined training.

Output JSONs use AVHBench schema with an added 'video_root' field pointing
to the absolute path of the directory containing the mp4 files.

Writes:
  data/avhbench_split_train.json   (~4,240 items, ~1,673 videos)
  data/avhbench_split_test.json    (~1,060 items, ~419 videos)
"""
from __future__ import annotations
import json
import random
from pathlib import Path

HOME = Path.home()
QA_PATH = HOME / "SOULEIMAN_repo" / "datasets" / "AVHBench" / "data" / "AVHBench_v0" / "json" / "qa.json"
VIDEO_ROOT = HOME / "SOULEIMAN_repo" / "datasets" / "AVHBench" / "data" / "AVHBench_v0" / "video"
OUT_DIR = Path("data")

TRAIN_FRACTION = 0.80
SEED = 42

YES_NO_TASKS = {
    "Audio-driven Video Hallucination",
    "Video-driven Audio Hallucination",
    "AV Matching",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(QA_PATH) as f:
        all_items = json.load(f)
    print(f"Loaded {len(all_items)} total AVHBench items")

    yn_items = [d for d in all_items if d["task"] in YES_NO_TASKS and d.get("label") in ("Yes", "No")]
    print(f"  Yes/no items: {len(yn_items)}")

    videos = sorted(set(d["video_id"] for d in yn_items))
    print(f"  Unique videos with yes/no items: {len(videos)}")

    rng = random.Random(SEED)
    rng.shuffle(videos)
    n_train = int(len(videos) * TRAIN_FRACTION)
    train_videos = set(videos[:n_train])
    test_videos = set(videos[n_train:])
    print(f"  Train videos: {len(train_videos)}  Test videos: {len(test_videos)}")

    train_items = []
    test_items = []
    for d in yn_items:
        d2 = dict(d)
        d2["video_root"] = str(VIDEO_ROOT)
        if d["video_id"] in train_videos:
            train_items.append(d2)
        else:
            test_items.append(d2)

    print(f"  Train items: {len(train_items)}  Test items: {len(test_items)}")

    train_out = OUT_DIR / "avhbench_split_train.json"
    test_out = OUT_DIR / "avhbench_split_test.json"
    with open(train_out, "w") as f:
        json.dump(train_items, f)
    with open(test_out, "w") as f:
        json.dump(test_items, f)
    print(f"Wrote {train_out}")
    print(f"Wrote {test_out}")


if __name__ == "__main__":
    main()