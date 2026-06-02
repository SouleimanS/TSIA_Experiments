"""Evaluate AVModelV6 on AVHBench yes/no judgment tasks.
Reuses forward_generate from av_model_v6 directly.

Usage:
    python -m av_ib.eval.v6_avhbench_eval \
        --ckpt-path runs/v6_tier1_b/step_1000.pt \
        --variant b --out-csv results/avhbench_tier1b.csv

For baseline (no checkpoint):
    python -m av_ib.eval.v6_avhbench_eval \
        --baseline --out-csv results/avhbench_baseline.csv
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from collections import defaultdict, Counter
from pathlib import Path
import torch

HOME        = Path.home()
AVH_ROOT    = HOME / "SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0"
AVH_QA      = AVH_ROOT / "json/qa.json"
AVH_VIDS    = AVH_ROOT / "video"
TASKS       = {"Audio-driven Video Hallucination",
               "Video-driven Audio Hallucination", "AV Matching"}

def parse_yn(text):
    h = text.strip().lower()[:40]
    if re.search(r"\byes\b", h): return "Yes"
    if re.search(r"\bno\b",  h): return "No"
    return "??"

def metrics(recs):
    n = len(recs)
    if not n: return {"n": 0}
    tp = sum(1 for r in recs if r[0]=="Yes" and r[1]=="Yes")
    tn = sum(1 for r in recs if r[0]=="No"  and r[1]=="No")
    fp = sum(1 for r in recs if r[0]=="No"  and r[1]=="Yes")
    fn = sum(1 for r in recs if r[0]=="Yes" and r[1]=="No")
    acc = (tp+tn)/n
    p_  = tp/(tp+fp) if (tp+fp) else 0.
    r_  = tp/(tp+fn) if (tp+fn) else 0.
    f1  = 2*p_*r_/(p_+r_) if (p_+r_) else 0.
    return {"n":n,"acc":acc,"f1":f1,"yes_pct":(tp+fp)/n*100,
            "TP":tp,"TN":tn,"FP":fp,"FN":fn,
            "non_yes_no":sum(1 for r in recs if r[1]=="??")}

def build_model(args):
    if args.baseline:
        print("[model] vanilla Qwen3-Omni (no ckpt)", flush=True)
        from av_ib.model.av_model_v6 import AVModelV6
        model = AVModelV6(use_lora=False, variant="b")
    else:
        print(f"[model] AVModelV6 variant={args.variant}", flush=True)
        from av_ib.model.av_model_v6 import AVModelV6
        model = AVModelV6(use_lora=True, variant=args.variant)
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("trainable_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  loaded {len(state)-len(unexpected)}/{len(state)} tensors "
              f"(unexpected={len(unexpected)})", flush=True)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--variant",   choices=("a","b","c"), default="b")
    ap.add_argument("--baseline",  action="store_true")
    ap.add_argument("--qa-json",   default=str(AVH_QA))
    ap.add_argument("--video-dir", default=str(AVH_VIDS))
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out-csv",   required=True)
    ap.add_argument("--out-json",  default=None)
    ap.add_argument("--video-vib", choices=("sink","standard"), default="sink")
    ap.add_argument("--fusion",    choices=("mutual","sink_sym"), default="mutual")
    args = ap.parse_args()

    with open(args.qa_json) as f:
        items = [d for d in json.load(f) if d["task"] in TASKS]
    print(f"Loaded {len(items)} judgment items")
    print("  Tasks:",  dict(Counter(d["task"]  for d in items)))
    print("  Labels:", dict(Counter(d["label"] for d in items)))
    if args.max_items:
        items = items[:args.max_items]

    model = build_model(args)
    video_dir = Path(args.video_dir)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    by_task, all_r, n_fail = defaultdict(list), [], 0
    t0 = time.time()

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","video_id","task","text","label","raw_pred","pred"])
        for i, rec in enumerate(items):
            vid = str(video_dir / f"{rec['video_id']}.mp4")
            try:
                with torch.no_grad():
                    raw = model.forward_generate([vid], [vid], [rec["text"]])[0]
                pred = parse_yn(raw)
            except Exception as e:
                raw, pred, n_fail = f"<ERR:{type(e).__name__}:{str(e)[:60]}>", "??", n_fail+1
                if i < 3:
                    import traceback; traceback.print_exc()
            w.writerow([i, rec["video_id"], rec["task"], rec["text"],
                        rec["label"], raw, pred])
            f.flush()
            by_task[rec["task"]].append((rec["label"], pred))
            all_r.append((rec["label"], pred))
            if (i+1) % 50 == 0:
                el = time.time()-t0
                print(f"  [{i+1}/{len(items)}] {(i+1)/el:.2f}/s  "
                      f"eta={int((len(items)-i-1)/((i+1)/el))}s  "
                      f"failed={n_fail}", flush=True)

    ov = metrics(all_r)
    res = {"n_items":len(items),"n_failed":n_fail,
           "elapsed_s":time.time()-t0,"overall":ov,
           "per_task":{t:metrics(r) for t,r in by_task.items()}}
    out_json = args.out_json or args.out_csv.replace(".csv",".json")
    Path(out_json).write_text(json.dumps(res, indent=2))

    print(f"\n=== AVHBench Results ===")
    print(f"  Overall  acc={ov['acc']*100:.1f}%  f1={ov['f1']:.3f}  "
          f"yes%={ov['yes_pct']:.1f}  non_yn={ov['non_yes_no']}  n={ov['n']}")
    for task, m in res["per_task"].items():
        print(f"  [{task[:40]}]  acc={m['acc']*100:.1f}%  f1={m['f1']:.3f}  n={m['n']}")
    print(f"  CSV:  {args.out_csv}")
    print(f"  JSON: {out_json}")
    print(f"Done in {res['elapsed_s']:.0f}s  failed={n_fail}", flush=True)

if __name__ == "__main__":
    main()
