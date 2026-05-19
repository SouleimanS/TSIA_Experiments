"""Sanity-check the new full_val_eval function against the known-good 1-epoch checkpoint.
Expected: ~0.68 (matches what eval_best_full_val.py reported earlier on the same checkpoint).
If it reports ~0.50, the new eval is buggy.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from av_ib.model.av_model_v1 import AVModelV1
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_avqa_3ep import full_val_eval, fix_records

HOME = Path.home()
ANN_VAL = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_val.json"
VID_ROOT = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "videos" / "Train"
CKPT = Path("runs/avqa_v1_1ep/best.pt")

print(f"Building model + loading {CKPT}", flush=True)
model = AVModelV1(use_lora=True).cuda()
sd = torch.load(CKPT, map_location="cuda")
own = dict(model.named_parameters())
n_loaded = 0
for k, v in sd.items():
    if k in own:
        own[k].data.copy_(v.data)
        n_loaded += 1
print(f"Loaded {n_loaded} params (of {len(sd)} in ckpt)", flush=True)

with open(ANN_VAL) as f:
    val_records = fix_records(json.load(f))
print(f"Running full_val_eval on {len(val_records)} items...", flush=True)
ev = full_val_eval(model, val_records, VID_ROOT, "cuda", -1)
print(f"\nAccuracy:  {ev['accuracy']:.4f}  ({ev['n_correct']} / {ev['n_total']})")
print(f"Failed:    {ev['n_failed']}")
print(f"Confusion: Y/Y={ev['confusion_YesYes']} Y/N={ev['confusion_YesNo']} "
      f"N/Y={ev['confusion_NoYes']} N/N={ev['confusion_NoNo']}")
print(f"Time:      {ev['eval_time_s']:.0f}s")
print(f"\nExpected: 0.68 (from eval_best_full_val.py)")
print(f"Got:      {ev['accuracy']:.4f}")
if abs(ev['accuracy'] - 0.68) < 0.03:
    print("→ EVAL FUNCTION IS FINE. 3-epoch model is genuinely collapsed.")
else:
    print("→ EVAL FUNCTION IS BUGGY. 3-epoch model may actually be learning.")
