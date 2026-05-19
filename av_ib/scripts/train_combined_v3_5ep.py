"""5-epoch single-GPU training on combined AVQA+AVHBench yes/no.

Trains v1 (no fusion, no VIB) on data/combined_train.json, evaluates on
data/avhbench_split_test.json every 300 steps.
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
from torch.utils.data import DataLoader

from av_ib.model.av_model_v3 import AVModelV3
from av_ib.data.avqa import AVQADataset
from av_ib.data.dummy import av_collate
from av_ib.eval.avhbench import _load_video, _load_audio, _build_prompt


HOME = Path.home()
ANN_TRAIN = Path("data/combined_train.json")
ANN_TEST = Path("data/avhbench_split_test.json")
AVQA_VID_ROOT = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "videos" / "Train"
OUT_DIR = Path("runs/combined_v3_5ep")

BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
NUM_EPOCHS = 5
EVAL_EVERY = 300
NUM_WORKERS = 4
DEVICE = "cuda"
BETA = 1e-3


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def avhbench_val_eval(model, val_records, device, step):
    model.eval()
    n_correct = n_total = n_failed = 0
    confusion = Counter()
    per_task = {}
    t0 = time.time()
    for rec in val_records:
        try:
            video_path = Path(rec["video_root"]) / f"{rec['video_id']}.mp4"
            videos = _load_video(str(video_path), device)
            audios = _load_audio(str(video_path), device)
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
            task = rec["task"]
            if task not in per_task:
                per_task[task] = [0, 0]
            per_task[task][1] += 1
            if pred_label == rec["label"]:
                per_task[task][0] += 1
            n_total += 1
        except Exception:
            n_failed += 1
    model.train()
    return {
        "step": step,
        "accuracy": n_correct / max(n_total, 1),
        "n_correct": n_correct,
        "n_total": n_total,
        "n_failed": n_failed,
        "eval_time_s": time.time() - t0,
        "confusion_YesYes": confusion.get(("Yes", "Yes"), 0),
        "confusion_YesNo":  confusion.get(("Yes", "No"),  0),
        "confusion_NoYes":  confusion.get(("No",  "Yes"), 0),
        "confusion_NoNo":   confusion.get(("No",  "No"),  0),
        "per_task": {k: {"correct": v[0], "total": v[1], "acc": v[0]/max(v[1],1)} for k, v in per_task.items()},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANN_TEST) as f:
        val_records = json.load(f)
    print(f"Val records (AVHBench held-out): {len(val_records)}", flush=True)

    print("=== Building AVModelV3 (VIB only, beta=1e-3) ===", flush=True)
    model = AVModelV3(use_lora=True).to(DEVICE)
    s = model.trainable_summary()
    print(f"  Trainable: {s['__total_trainable__']:,}", flush=True)

    print("\n=== Building train dataloader (combined AVQA + AVHBench) ===", flush=True)
    train_ds = AVQADataset(ann_path=ANN_TRAIN, video_root=AVQA_VID_ROOT)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=av_collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"  {len(train_ds)} items, batch={BATCH_SIZE}, {steps_per_epoch} steps/epoch, total={total_steps}", flush=True)

    optim = torch.optim.AdamW(
        trainable_params(model),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    train_log = open(OUT_DIR / "train_log.jsonl", "w")
    eval_log = open(OUT_DIR / "eval_log.jsonl", "w")

    print(f"\n=== Training {NUM_EPOCHS} epochs, eval every {EVAL_EVERY} on AVHBench held-out ===\n", flush=True)
    model.train()
    t0 = time.time()
    best_acc = 0.0
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        for batch in train_dl:
            videos = batch["videos"].to(DEVICE, non_blocking=True)
            audio_mels = batch["audio_mels"].to(DEVICE, non_blocking=True)
            nll, kl = model.forward_train(videos, audio_mels, batch["prompts"], batch["answers"])
            loss = nll + BETA * kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params(model), GRAD_CLIP)
            optim.step()

            rec = {
                "step": global_step,
                "epoch": epoch,
                "loss": float(loss.item()),
                "grad_norm": float(gnorm),
                "elapsed_s": time.time() - t0,
            }
            train_log.write(json.dumps(rec) + "\n")
            train_log.flush()

            if global_step % 10 == 0:
                print(f"  step {global_step:4d} (ep{epoch})  loss={rec['loss']:.4f}  "
                      f"gnorm={rec['grad_norm']:.2f}  elapsed={rec['elapsed_s']:.0f}s", flush=True)

            if (global_step + 1) % EVAL_EVERY == 0 or global_step == total_steps - 1:
                print(f"\n  ---- Eval at step {global_step} ----", flush=True)
                ev = avhbench_val_eval(model, val_records, DEVICE, global_step)
                print(f"  acc={ev['accuracy']:.4f} ({ev['n_correct']}/{ev['n_total']})  "
                      f"failed={ev['n_failed']}  time={ev['eval_time_s']:.0f}s", flush=True)
                print(f"  confusion: Y/Y={ev['confusion_YesYes']} Y/N={ev['confusion_YesNo']} "
                      f"N/Y={ev['confusion_NoYes']} N/N={ev['confusion_NoNo']}", flush=True)
                for task, m in ev["per_task"].items():
                    print(f"    {task}: {m['acc']:.4f} ({m['correct']}/{m['total']})", flush=True)
                print(flush=True)
                eval_log.write(json.dumps(ev) + "\n")
                eval_log.flush()
                if ev["accuracy"] > best_acc:
                    best_acc = ev["accuracy"]
                    torch.save(
                        {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                        OUT_DIR / "best.pt",
                    )
                    print(f"  saved best.pt (acc={best_acc:.4f})\n", flush=True)

            global_step += 1

    torch.save(
        {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        OUT_DIR / "final.pt",
    )
    train_log.close()
    eval_log.close()
    print(f"\n=== Done. Total: {time.time()-t0:.0f}s. Best acc: {best_acc:.4f} ===", flush=True)


if __name__ == "__main__":
    main()