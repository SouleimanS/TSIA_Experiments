"""3-epoch single-GPU training on AVQA yes/no for AVModelV2 (Fusion only).

Identical to train_avqa_3ep.py but instantiates AVModelV2 and writes to
runs/avqa_v2_3ep/.
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

from av_ib.model.av_model_v2 import AVModelV2
from av_ib.data.avqa import AVQADataset
from av_ib.data.dummy import av_collate
from av_ib.eval.avhbench import _load_video, _load_audio, _build_prompt


HOME = Path.home()
ANN_TRAIN = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_train.json"
ANN_VAL   = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_val.json"
VID_ROOT  = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "videos" / "Train"
# OUT_DIR is now a CLI arg

BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
NUM_EPOCHS = 3
EVAL_EVERY = 300
NUM_WORKERS = 4
DEVICE = "cuda"


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def fix_records(records):
    out = []
    for r in records:
        r = dict(r)
        r["video_id"] = re.sub(r"_\d{6}$", "", r["video_id"])
        out.append(r)
    return out


def full_val_eval(model, val_records, video_root, device, step):
    model.eval()
    n_correct = n_total = n_failed = 0
    confusion = Counter()
    t0 = time.time()
    for i, rec in enumerate(val_records):
        try:
            videos = _load_video(str(video_root / f"{rec['video_id']}.mp4"), device)
            audios = _load_audio(str(video_root / f"{rec['video_id']}.mp4"), device)
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
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=1, help="Cross-attention block depth")
    ap.add_argument("--output-dir", type=str, required=True, help="runs/<name>/")
    ap.add_argument("--smoke-steps", type=int, default=None,
                    help="If set, run only N training steps then exit (no eval).")
    args = ap.parse_args()
    N_BLOCKS = args.n_blocks
    OUT_DIR = Path(args.output_dir)
    SMOKE_STEPS = args.smoke_steps
    print(f"=== Config: n_blocks={N_BLOCKS}, output_dir={OUT_DIR}, smoke_steps={SMOKE_STEPS} ===", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANN_VAL) as f:
        val_records = fix_records(json.load(f))
    print(f"Val records: {len(val_records)}", flush=True)

    print("=== Building AVModelV2 (fusion only) ===", flush=True)
    model = AVModelV2(use_lora=True, fusion_n_blocks=N_BLOCKS).to(DEVICE)
    s = model.trainable_summary()
    print(f"  Trainable: {s['__total_trainable__']:,}", flush=True)
    print(f"  Fusion adds: {s.get('fusion', 0):,} params", flush=True)

    print("\n=== Building train dataloader ===", flush=True)
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

    print(f"\n=== Training {NUM_EPOCHS} epochs, eval every {EVAL_EVERY} on full val ===\n", flush=True)
    model.train()
    t0 = time.time()
    best_acc = 0.0
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        for batch in train_dl:
            videos = batch["videos"].to(DEVICE, non_blocking=True)
            audio_mels = batch["audio_mels"].to(DEVICE, non_blocking=True)
            loss = model.forward_train(videos, audio_mels, batch["prompts"], batch["answers"])

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
                ev = full_val_eval(model, val_records, VID_ROOT, DEVICE, global_step)
                print(f"  acc={ev['accuracy']:.4f} ({ev['n_correct']}/{ev['n_total']})  "
                      f"failed={ev['n_failed']}  time={ev['eval_time_s']:.0f}s", flush=True)
                print(f"  confusion: Y/Y={ev['confusion_YesYes']} Y/N={ev['confusion_YesNo']} "
                      f"N/Y={ev['confusion_NoYes']} N/N={ev['confusion_NoNo']}\n", flush=True)
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
            if SMOKE_STEPS is not None and global_step >= SMOKE_STEPS:
                print(f"\n=== SMOKE OK: {SMOKE_STEPS} steps completed for n_blocks={N_BLOCKS} ===", flush=True)
                train_log.close()
                eval_log.close()
                return

    torch.save(
        {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        OUT_DIR / "final.pt",
    )
    train_log.close()
    eval_log.close()
    print(f"\n=== Done. Total: {time.time()-t0:.0f}s. Best acc: {best_acc:.4f} ===", flush=True)


if __name__ == "__main__":
    main()