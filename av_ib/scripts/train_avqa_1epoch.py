"""Single-GPU 1-epoch training on AVQA yes/no, with periodic val eval.

Trains AVModelV1 on the 11,010 train items, logs loss every step, runs eval
on a 200-item val subset every 200 steps, saves checkpoint at end.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Subset

from av_ib.model.av_model_v1 import AVModelV1
from av_ib.data.avqa import AVQADataset
from av_ib.data.dummy import av_collate
from av_ib.eval.avhbench import run_eval as run_eval_csv


# Paths
HOME = Path.home()
ANN_TRAIN = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_train.json"
ANN_VAL   = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_val.json"
VID_ROOT  = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "videos" / "Train"

OUT_DIR = Path("runs/avqa_v1_1ep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Hyperparameters (one place)
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
EVAL_EVERY = 200          # steps
EVAL_SUBSET = 200         # number of val items per eval
NUM_WORKERS = 4
DEVICE = "cuda"


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def quick_val_accuracy(model, val_subset_records, video_root, device):
    """Generate on a small subset, parse Yes/No, compute accuracy.

    val_subset_records: list of {video_id, task, text, label} dicts.
    Returns (accuracy, n_correct, n_total, n_failed).
    """
    import re
    from av_ib.eval.avhbench import _load_video, _load_audio, _build_prompt

    model.eval()
    n_correct = n_total = n_failed = 0
    for rec in val_subset_records:
        video_path = video_root / f"{rec['video_id']}.mp4"
        try:
            videos = _load_video(str(video_path), device)
            audios = _load_audio(str(video_path), device)
            prompt = _build_prompt(rec["task"], rec["text"])
            with torch.no_grad():
                pred = model.forward_generate(videos, audios, [prompt], max_new_tokens=10)[0]
            pred_clean = pred.strip().lower()
            # Match "yes"/"no" anywhere in first 30 chars of generation
            head = pred_clean[:30]
            if re.search(r"\byes\b", head):
                pred_label = "Yes"
            elif re.search(r"\bno\b", head):
                pred_label = "No"
            else:
                pred_label = "??"
            if pred_label == rec["label"]:
                n_correct += 1
            n_total += 1
        except Exception:
            n_failed += 1
    model.train()
    acc = n_correct / max(n_total, 1)
    return acc, n_correct, n_total, n_failed


def main():
    # Strip the _NNNNNN suffix from video_id so they map to files on disk.
    # (AVHBench's run_eval expects video_id to equal the filename stem.)
    import re
    def fix_records(records):
        out = []
        for r in records:
            r = dict(r)
            r["video_id"] = re.sub(r"_\d{6}$", "", r["video_id"])
            out.append(r)
        return out

    print("=== Loading val records (subset for periodic eval) ===")
    with open(ANN_VAL) as f:
        val_records_full = fix_records(json.load(f))
    val_subset = val_records_full[:EVAL_SUBSET]
    print(f"  val subset: {len(val_subset)} items (of {len(val_records_full)} total)")

    print("\n=== Building AVModelV1 ===")
    model = AVModelV1(use_lora=True).to(DEVICE)
    summary = model.trainable_summary()
    print(f"  Trainable: {summary['__total_trainable__']:,}")

    print("\n=== Building train dataloader ===")
    train_ds = AVQADataset(ann_path=ANN_TRAIN, video_root=VID_ROOT)
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
    print(f"  {len(train_ds)} items, batch={BATCH_SIZE}, {steps_per_epoch} steps/epoch")

    # Optimizer
    optim = torch.optim.AdamW(
        trainable_params(model),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # Logging
    train_log = open(OUT_DIR / "train_log.jsonl", "w")
    eval_log = open(OUT_DIR / "eval_log.jsonl", "w")

    print(f"\n=== Training 1 epoch ({steps_per_epoch} steps), eval every {EVAL_EVERY} ===\n")
    model.train()
    t0 = time.time()
    best_acc = 0.0

    for step, batch in enumerate(train_dl):
        videos = batch["videos"].to(DEVICE, non_blocking=True)
        audio_mels = batch["audio_mels"].to(DEVICE, non_blocking=True)
        loss = model.forward_train(videos, audio_mels, batch["prompts"], batch["answers"])

        optim.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable_params(model), GRAD_CLIP)
        optim.step()

        rec = {
            "step": step,
            "loss": float(loss.item()),
            "grad_norm": float(gnorm),
            "elapsed_s": time.time() - t0,
        }
        train_log.write(json.dumps(rec) + "\n")
        train_log.flush()

        if step % 10 == 0 or step == steps_per_epoch - 1:
            print(f"  step {step:4d}/{steps_per_epoch}  "
                  f"loss={rec['loss']:.4f}  gnorm={rec['grad_norm']:.2f}  "
                  f"elapsed={rec['elapsed_s']:.0f}s")

        # Periodic eval
        if (step + 1) % EVAL_EVERY == 0 or step == steps_per_epoch - 1:
            print(f"\n  ---- Eval at step {step} on {len(val_subset)} val items ----")
            t_eval = time.time()
            acc, c, n, fail = quick_val_accuracy(model, val_subset, VID_ROOT, DEVICE)
            eval_rec = {
                "step": step,
                "val_subset_n": n,
                "val_subset_correct": c,
                "val_subset_accuracy": acc,
                "val_subset_failed": fail,
                "eval_time_s": time.time() - t_eval,
            }
            eval_log.write(json.dumps(eval_rec) + "\n")
            eval_log.flush()
            print(f"  val acc: {acc:.3f} ({c}/{n})  failed={fail}  eval_time={eval_rec['eval_time_s']:.0f}s\n")

            if acc > best_acc:
                best_acc = acc
                torch.save(
                    {n: p for n, p in model.named_parameters() if p.requires_grad},
                    OUT_DIR / "best.pt",
                )
                print(f"  saved best.pt (acc={acc:.3f})\n")

    train_log.close()
    eval_log.close()

    # Final checkpoint
    torch.save(
        {n: p for n, p in model.named_parameters() if p.requires_grad},
        OUT_DIR / "final.pt",
    )
    print(f"\n=== Done. Total: {time.time()-t0:.0f}s, best val acc: {best_acc:.3f} ===")


if __name__ == "__main__":
    main()
