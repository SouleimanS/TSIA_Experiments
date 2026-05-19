"""3-epoch DDP training on AVQA yes/no with rank-0 eval.

Launch with:
    torchrun --standalone --nproc_per_node=8 scripts/train_avqa_3ep_ddp.py
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from av_ib.model.av_model_v1 import AVModelV1
from av_ib.data.avqa import AVQADataset
from av_ib.data.dummy import av_collate
from av_ib.eval.avhbench import _load_video, _load_audio, _build_prompt


HOME = Path.home()
ANN_TRAIN = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_train.json"
ANN_VAL   = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "AVQA" / "AVQA_dataset" / "avhbench_val.json"
VID_ROOT  = HOME / "SOULEIMAN_repo" / "datasets" / "AVQA" / "videos" / "Train"
OUT_DIR   = Path("runs/avqa_v1_3ep_ddp")

PER_GPU_BATCH = 8
LR = 5e-5
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
NUM_EPOCHS = 3
EVAL_EVERY = 100
NUM_WORKERS = 4


def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def rprint(*args, **kw):
    if is_rank0():
        print(*args, **kw, flush=True)


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def load_val_records():
    with open(ANN_VAL) as f:
        records = json.load(f)
    for r in records:
        r["video_id"] = re.sub(r"_\d{6}$", "", r["video_id"])
    return records


def run_full_val_eval(model, records, video_root, device, step):
    from collections import Counter
    model.eval()
    underlying = model.module if isinstance(model, DDP) else model
    n_correct = n_total = n_failed = 0
    confusion = Counter()
    t0 = time.time()
    for i, rec in enumerate(records):
        try:
            videos = _load_video(str(video_root / f"{rec['video_id']}.mp4"), device)
            audios = _load_audio(str(video_root / f"{rec['video_id']}.mp4"), device)
            prompt = _build_prompt(rec["task"], rec["text"])
            with torch.no_grad():
                pred = underlying.forward_generate(videos, audios, [prompt], max_new_tokens=10)[0]
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
        if (i + 1) % 200 == 0:
            rprint(f"    eval[{step}] {i+1}/{len(records)}  acc={n_correct/max(n_total,1):.3f}")
    model.train()
    acc = n_correct / max(n_total, 1)
    return {
        "step": step,
        "accuracy": acc,
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
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = f"cuda:{local_rank}"

    rprint(f"=== DDP world_size={world_size}, rank={rank} on {device} ===")
    rprint(f"=== Per-GPU batch={PER_GPU_BATCH}, effective batch={PER_GPU_BATCH*world_size} ===")

    if is_rank0():
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    rprint("\n=== Building AVModelV1 ===")
    model = AVModelV1(use_lora=True).to(device)
    if is_rank0():
        s = model.trainable_summary()
        rprint(f"  Trainable params: {s['__total_trainable__']:,}")

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
    )

    rprint("\n=== Building train dataloader ===")
    train_ds = AVQADataset(ann_path=ANN_TRAIN, video_root=VID_ROOT)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=PER_GPU_BATCH,
        sampler=sampler,
        collate_fn=av_collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(train_dl)
    rprint(f"  train items: {len(train_ds)}, per-GPU steps/epoch: {steps_per_epoch}, total: {steps_per_epoch*NUM_EPOCHS}")

    val_records = load_val_records() if is_rank0() else None

    optim = torch.optim.AdamW(
        trainable_params(model),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    train_log = open(OUT_DIR / "train_log.jsonl", "w") if is_rank0() else None
    eval_log  = open(OUT_DIR / "eval_log.jsonl",  "w") if is_rank0() else None

    rprint(f"\n=== Training {NUM_EPOCHS} epochs ===\n")
    model.train()
    t0 = time.time()
    best_acc = 0.0
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        for batch in train_dl:
            videos     = batch["videos"].to(device, non_blocking=True)
            audio_mels = batch["audio_mels"].to(device, non_blocking=True)
            loss = model.module.forward_train(videos, audio_mels, batch["prompts"], batch["answers"])

            optim.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params(model), GRAD_CLIP)
            optim.step()

            loss_t = loss.detach().clone()
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)

            if is_rank0():
                rec = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss_t.item()),
                    "grad_norm": float(gnorm),
                    "elapsed_s": time.time() - t0,
                }
                train_log.write(json.dumps(rec) + "\n")
                train_log.flush()
                if global_step % 10 == 0:
                    rprint(f"  step {global_step:4d} (ep{epoch})  "
                           f"loss={rec['loss']:.4f}  gnorm={rec['grad_norm']:.2f}  "
                           f"elapsed={rec['elapsed_s']:.0f}s")

            if (global_step + 1) % EVAL_EVERY == 0:
                if is_rank0():
                    rprint(f"\n  ---- Eval at step {global_step} ----")
                    ev = run_full_val_eval(model, val_records, VID_ROOT, device, global_step)
                    rprint(f"  acc={ev['accuracy']:.4f} ({ev['n_correct']}/{ev['n_total']})  "
                           f"failed={ev['n_failed']}  time={ev['eval_time_s']:.0f}s")
                    rprint(f"  confusion: Y/Y={ev['confusion_YesYes']} Y/N={ev['confusion_YesNo']} "
                           f"N/Y={ev['confusion_NoYes']} N/N={ev['confusion_NoNo']}\n")
                    eval_log.write(json.dumps(ev) + "\n")
                    eval_log.flush()
                    if ev["accuracy"] > best_acc:
                        best_acc = ev["accuracy"]
                        underlying = model.module
                        torch.save(
                            {n: p.detach().cpu() for n, p in underlying.named_parameters() if p.requires_grad},
                            OUT_DIR / "best.pt",
                        )
                        rprint(f"  saved best.pt (acc={best_acc:.4f})\n")
                dist.barrier()

            global_step += 1

    if is_rank0():
        underlying = model.module
        torch.save(
            {n: p.detach().cpu() for n, p in underlying.named_parameters() if p.requires_grad},
            OUT_DIR / "final.pt",
        )
        rprint(f"\n=== Final eval ===")
        ev = run_full_val_eval(model, val_records, VID_ROOT, device, global_step)
        rprint(f"  final acc={ev['accuracy']:.4f}")
        eval_log.write(json.dumps(ev) + "\n")
        train_log.close()
        eval_log.close()
        rprint(f"\n=== Done. Total time: {time.time()-t0:.0f}s.  Best acc: {best_acc:.4f} ===")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
