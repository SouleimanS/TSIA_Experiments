"""Combine AVQA training items with AVHBench split-train items.

AVQA items: as-is (no video_root field, dataset class strips _NNNNNN suffix
            and uses its default video_root for AVQA videos).
AVHBench items: have explicit video_root field pointing to AVHBench videos.

Writes:
  data/combined_train.json  (~15,250 items)
"""
from __future__ import annotations
import json
import random
from pathlib import Path

HOME = Path.home()
AVQA_TRAIN = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_train.json"
AVHBENCH_SPLIT_TRAIN = Path("data/avhbench_split_train.json")
OUT_PATH = Path("data/combined_train.json")
SEED = 42


def main():
    with open(AVQA_TRAIN) as f:
        avqa = json.load(f)
    print(f"AVQA items: {len(avqa)}")

    with open(AVHBENCH_SPLIT_TRAIN) as f:
        avh = json.load(f)
    print(f"AVHBench split-train items: {len(avh)}")

    # AVQA items have no 'video_root' (default behavior).
    # AVHBench items already have 'video_root' from the split script.
    combined = list(avqa) + list(avh)
    print(f"Combined: {len(combined)}")

    # Shuffle so the dataloader sees a mix of both throughout training.
    rng = random.Random(SEED)
    rng.shuffle(combined)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(combined, f)
    print(f"Wrote {OUT_PATH}")

    # Sanity: count how many of each
    n_avq = sum(1 for d in combined if "video_root" not in d)
    n_avh = sum(1 for d in combined if "video_root" in d)
    print(f"Sanity: {n_avq} AVQA + {n_avh} AVHBench = {n_avq + n_avh}")


if __name__ == "__main__":
    main()