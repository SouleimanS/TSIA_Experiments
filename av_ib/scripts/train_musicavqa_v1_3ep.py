"""3-epoch single-GPU training on MUSIC-AVQA for AVModelV1 (baseline)."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from av_ib.model.av_model_v1 import AVModelV1
from av_ib.data.musicavqa import MusicAVQADataset
from av_ib.data.dummy import av_collate
from av_ib.eval.musicavqa_eval import musicavqa_val_eval


HOME = Path.home()
MUSICAVQA_ROOT = HOME / "SOULEIMAN_repo" / "datasets" / "MUSIC-AVQA" / "MUSIC-AVQA" / "data" / "json_update"
ANN_TRAIN = MUSICAVQA_ROOT / "avqa-train.json"
ANN_VAL = MUSICAVQA_ROOT / "avqa-val.json"
VID_ROOT = HOME / "SOULEIMAN_repo" / "datasets" / "MUSIC-AVQA" / "videos" / "all"

BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
NUM_EPOCHS = 3
EVAL_EVERY = 1000
VAL_SUBSAMPLE = 500
NUM_WORKERS = 4
DEVICE = "cuda"


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--smoke-steps", type=int, default=None)
    args = ap.parse_args()
    OUT_DIR = Path(args.output_dir)
    SMOKE_STEPS = args.smoke_steps
    print(f"=== Config: output_dir={OUT_DIR}, smoke_steps={SMOKE_STEPS} ===", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANN_VAL) as f:
        val_records = [r for r in json.load(f) if r.get("question_deleted", 0) == 0]
    print(f"Val records: {len(val_records)}", flush=True)

    print("=== Building AVModelV1 (baseline) ===", flush=True)
    model = AVModelV1(use_lora=True).to(DEVICE)
    s = model.trainable_summary()
    print(f"  Trainable: {s['__total_trainable__']:,}", flush=True)

    print("\n=== Building train dataloader ===", flush=True)
    train_ds = MusicAVQADataset(ann_path=ANN_TRAIN, video_root=VID_ROOT)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=av_collate,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"  {len(train_ds)} items, batch={BATCH_SIZE}, "
          f"{steps_per_epoch} steps/epoch, total={total_steps}", flush=True)

    optim = torch.optim.AdamW(
        trainable_params(model), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
    )

    train_log = open(OUT_DIR / "train_log.jsonl", "w")
    eval_log = open(OUT_DIR / "eval_log.jsonl", "w")

    print(f"\n=== Training {NUM_EPOCHS} epochs, eval every {EVAL_EVERY} ===\n", flush=True)
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
                "step": global_step, "epoch": epoch, "loss": float(loss.item()),
                "grad_norm": float(gnorm), "elapsed_s": time.time() - t0,
            }
            train_log.write(json.dumps(rec) + "\n")
            train_log.flush()

            if global_step % 10 == 0:
                print(f"  step {global_step:5d} (ep{epoch})  loss={rec['loss']:.4f}  "
                      f"gnorm={rec['grad_norm']:.2f}  elapsed={rec['elapsed_s']:.0f}s", flush=True)

            is_final = (global_step == total_steps - 1)
            if (global_step + 1) % EVAL_EVERY == 0 or is_final:
                print(f"\n  ---- Eval at step {global_step} "
                      f"({'FULL' if is_final else f'subsample={VAL_SUBSAMPLE}'}) ----", flush=True)
                ev = musicavqa_val_eval(
                    model, val_records, VID_ROOT, DEVICE, global_step,
                    max_records=None if is_final else VAL_SUBSAMPLE,
                )
                print(f"  acc={ev['accuracy']:.4f} ({ev['n_correct']}/{ev['n_total']})  "
                      f"??={ev['non_match']}  failed={ev['n_failed']}  "
                      f"missing_vid={ev['n_missing_video']}  time={ev['eval_time_s']:.0f}s", flush=True)
                for mod, m in ev["per_modality"].items():
                    print(f"    {mod:14s} n={m['n']:4d}  acc={m['acc']:.4f}", flush=True)
                print()
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
                print(f"\n=== SMOKE OK: {SMOKE_STEPS} steps completed ===", flush=True)
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
