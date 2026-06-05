"""Evaluate v6 checkpoints (variant b) on AVHBench yes/no judgment tasks."""
from __future__ import annotations
import argparse, csv, json, re, time
from collections import defaultdict, Counter
from pathlib import Path
import torch

HOME = Path.home()
AVHBENCH_ROOT = HOME / "SOULEIMAN_repo/datasets/AVHBench/data/AVHBench_v0"
AVHBENCH_QA   = AVHBENCH_ROOT / "json/qa.json"
AVHBENCH_VIDS = AVHBENCH_ROOT / "video"
JUDGMENT_TASKS = {"Audio-driven Video Hallucination",
                  "Video-driven Audio Hallucination", "AV Matching"}

def parse_yes_no(text):
    h = text.strip().lower()[:40]
    if re.search(r"\byes\b", h): return "Yes"
    if re.search(r"\bno\b",  h): return "No"
    return "??"

def metrics(records):
    n = len(records)
    if not n: return {"n": 0}
    tp = sum(1 for r in records if r["label"]=="Yes" and r["pred"]=="Yes")
    tn = sum(1 for r in records if r["label"]=="No"  and r["pred"]=="No")
    fp = sum(1 for r in records if r["label"]=="No"  and r["pred"]=="Yes")
    fn = sum(1 for r in records if r["label"]=="Yes" and r["pred"]=="No")
    qq = sum(1 for r in records if r["pred"]=="??")
    acc  = (tp+tn)/n
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return {"n":n,"acc":acc,"precision":prec,"recall":rec,"f1":f1,
            "yes_pct":(tp+fp)/n*100,"TP":tp,"TN":tn,"FP":fp,"FN":fn,"non_yes_no":qq}

def load_av(video_path, device):
    # Reuse the same preprocessors as the v6 training pipeline
    from av_ib.eval.avhbench import _load_video, _load_audio
    return _load_video(video_path, device), _load_audio(video_path, device)

def build_v6(variant, ckpt_path, device):
    print(f"[1/3] Building AVModelV6 variant={variant}...", flush=True)
    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(use_lora=True, variant=variant).to(device)
    print(f"[2/3] Loading {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location=device)
    if "trainable_state" in sd: sd = sd["trainable_state"]
    own = dict(model.named_parameters())
    n_ok = sum(1 for k,v in sd.items() if k in own and not own[k].data.copy_(v.data) is None)
    print(f"    Loaded {n_ok}/{len(sd)} params", flush=True)
    return model.eval()

def build_baseline(device):
    print("[1/3] Loading vanilla Qwen3-Omni baseline...", flush=True)
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    model_path = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)
    # attach processor so infer() can reach it
    model._avhbench_processor = processor
    return model

def infer(model, videos, audios, text, is_baseline, video_path=None):
    if is_baseline:
        from av_ib.eval.qwen_omni_baseline import build_conversation
        from qwen_omni_utils import process_mm_info
        processor = model._avhbench_processor
        convo = build_conversation(video_path, text)
        text_input = processor.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
        audios_in, images_in, videos_in = process_mm_info(convo, use_audio_in_video=True)
        inputs = processor(text=text_input, audio=audios_in, images=images_in,
                           videos=videos_in, return_tensors="pt",
                           padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, use_audio_in_video=True)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]
    with torch.no_grad():
        return model.forward_generate(videos, audios, [text], max_new_tokens=20)[0]

def run_eval(model, items, video_dir, out_csv, out_json, is_baseline=False, every=20):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    f = open(out_csv, "w", newline="")
    w = csv.writer(f)
    w.writerow(["idx","video_id","task","text","label","raw_pred","pred"])
    by_task, all_r, n_fail, t0 = defaultdict(list), [], 0, time.time()
    print(f"[3/3] Evaluating {len(items)} items...", flush=True)
    for i, rec in enumerate(items):
        vid = video_dir / f"{rec['video_id']}.mp4"
        try:
            vs, au = load_av(str(vid), "cuda")
            raw = infer(model, vs, au, rec["text"], is_baseline, video_path=str(vid))
            pred = parse_yes_no(raw)
        except Exception as e:
            raw, pred, n_fail = f"<ERR:{e}>", "??", n_fail+1
        w.writerow([i, rec["video_id"], rec["task"], rec["text"], rec["label"], raw, pred])
        f.flush()
        entry = {"label": rec["label"], "pred": pred}
        by_task[rec["task"]].append(entry); all_r.append(entry)
        if (i+1) % every == 0:
            el = time.time()-t0
            print(f"  [{i+1}/{len(items)}] {(i+1)/el:.2f} ex/s  eta={int((len(items)-i-1)/((i+1)/el))}s  failed={n_fail}", flush=True)
    f.close()
    res = {"n_items":len(items),"n_failed":n_fail,"elapsed_s":time.time()-t0,
           "overall":metrics(all_r),
           "per_task":{t:metrics(r) for t,r in by_task.items()}}
    Path(out_json).write_text(json.dumps(res, indent=2))
    ov = res["overall"]
    print(f"\n=== Results ===")
    print(f"  Overall acc:  {ov['acc']*100:.1f}%  (n={ov['n']})")
    print(f"  Yes%: {ov['yes_pct']:.1f}%   non_yes_no: {ov['non_yes_no']}")
    for task, m in res["per_task"].items():
        print(f"  [{task[:38]}]  acc={m['acc']*100:.1f}%  f1={m['f1']:.3f}  n={m['n']}")
    print(f"Done in {res['elapsed_s']:.0f}s  failed={n_fail}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--variant",
                    choices=("a", "c",
                             "b_std_fusion", "b_topk_fusion",
                             "b_std_nofusion", "b_topk_nofusion",
                             "b_topk_fusion_adavib", "b_topk_fusion_adavib2"),
                    default="b_std_fusion")
    ap.add_argument("--baseline",  action="store_true")
    ap.add_argument("--qa-json",   default=str(AVHBENCH_QA))
    ap.add_argument("--video-dir", default=str(AVHBENCH_VIDS))
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out-csv",   required=True)
    ap.add_argument("--out-json",  required=True)
    args = ap.parse_args()
    with open(args.qa_json) as f: all_items = json.load(f)
    items = [d for d in all_items if d["task"] in JUDGMENT_TASKS]
    print(f"Loaded {len(items)} judgment items")
    print("  Tasks:", dict(Counter(d["task"] for d in items)))
    print("  Labels:", dict(Counter(d["label"] for d in items)))
    if args.max_items: items = items[:args.max_items]
    model = build_baseline("cuda") if args.baseline else build_v6(args.variant, args.ckpt_path, "cuda")
    run_eval(model, items, Path(args.video_dir), args.out_csv, args.out_json, args.baseline)

if __name__ == "__main__":
    main()
