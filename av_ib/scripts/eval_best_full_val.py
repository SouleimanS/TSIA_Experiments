"""Run full val eval on best.pt from the 1-epoch training run.

Reports accuracy on all 1238 val items, plus the yes/no confusion matrix
so we can see if the model is biased.
"""
from __future__ import annotations
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from av_ib.model.av_model_v1 import AVModelV1
from av_ib.eval.avhbench import _load_video, _load_audio, _build_prompt


HOME = Path.home()
ANN_VAL = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_val.json"
VID_ROOT = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "videos" / "Train"
CKPT = Path("runs/avqa_v1_1ep/best.pt")
OUT = Path("runs/avqa_v1_1ep/full_val_eval.csv")


def main():
    print("=== Building AVModelV1 ===")
    model = AVModelV1(use_lora=True).cuda()

    print(f"\n=== Loading checkpoint: {CKPT} ===")
    sd = torch.load(CKPT, map_location="cuda")
    # best.pt stored as {name: param}; load into model
    own = dict(model.named_parameters())
    missing, extra = [], []
    for k, v in sd.items():
        if k in own:
            own[k].data.copy_(v.data)
        else:
            extra.append(k)
    for k in own:
        if k not in sd:
            missing.append(k)
    print(f"  Loaded {len(sd) - len(extra)} params, missing={len(missing)}, extra={len(extra)}")

    print(f"\n=== Loading val records ===")
    with open(ANN_VAL) as f:
        records = json.load(f)
    # Strip _NNNNNN suffix so video_id matches filename stem
    for r in records:
        r["video_id"] = re.sub(r"_\d{6}$", "", r["video_id"])
    print(f"  {len(records)} val items")

    model.eval()
    n_correct = n_total = n_failed = 0
    confusion = Counter()  # (gold, pred) -> count
    t0 = time.time()

    with open(OUT, "w") as fout:
        fout.write("idx,video_id,gold,pred,text\n")
        for i, rec in enumerate(records):
            try:
                videos = _load_video(str(VID_ROOT / f"{rec['video_id']}.mp4"), "cuda")
                audios = _load_audio(str(VID_ROOT / f"{rec['video_id']}.mp4"), "cuda")
                prompt = _build_prompt(rec["task"], rec["text"])
                with torch.no_grad():
                    pred = model.forward_generate(videos, audios, [prompt], max_new_tokens=10)[0]
                head = pred.strip().lower()[:30]
                if re.search(r"\byes\b", head):
                    pred_label = "Yes"
                elif re.search(r"\bno\b", head):
                    pred_label = "No"
                else:
                    pred_label = "??"
                if pred_label == rec["label"]:
                    n_correct += 1
                confusion[(rec["label"], pred_label)] += 1
                n_total += 1
                fout.write(f'{i},{rec["video_id"]},{rec["label"]},{pred_label},"{rec["text"][:80]}"\n')
            except Exception as e:
                n_failed += 1
                fout.write(f'{i},{rec["video_id"]},{rec["label"]},ERROR,"{str(e)[:80]}"\n')

            if (i + 1) % 100 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(records) - i - 1) / rate
                acc = n_correct / max(n_total, 1)
                print(f"  [{i+1}/{len(records)}] acc={acc:.3f} ({n_correct}/{n_total}) "
                      f"failed={n_failed} rate={rate:.2f}/s eta={eta:.0f}s")

    el = time.time() - t0
    acc = n_correct / max(n_total, 1)
    print(f"\n=== Final ===")
    print(f"  Accuracy:  {acc:.4f}  ({n_correct} / {n_total})")
    print(f"  Failed:    {n_failed}")
    print(f"  Time:      {el:.0f}s ({len(records)/el:.2f} items/s)")
    print(f"\n=== Confusion matrix ===")
    print(f"             pred=Yes   pred=No   pred=??")
    for gold in ("Yes", "No"):
        row = "  gold={:<3}    {:>6}    {:>6}    {:>6}".format(
            gold,
            confusion.get((gold, "Yes"), 0),
            confusion.get((gold, "No"),  0),
            confusion.get((gold, "??"),  0),
        )
        print(row)

if __name__ == "__main__":
    main()
