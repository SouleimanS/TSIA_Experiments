#!/usr/bin/env python3
"""Evaluate Qwen3-Omni checkpoints on Music-AVQA.

Reports top-1 accuracy overall and per modality (Audio / Visual / Audio-Visual)
against the 41-token closed vocabulary.

── Single run ──────────────────────────────────────────────────────────────────
HF (transformers):
    python eval_musicavqa.py \
        --base-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --adapter    ./ckpts/musicavqa_1k/final \
        --ann-path   /data/musicavqa/json_update/avqa-test.json \
        --video-root /data/musicavqa/videos \
        --label      musicavqa_1k

vLLM (faster, recommended):
    python eval_musicavqa.py \
        --base-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --adapter    ./ckpts/musicavqa_1k/final \
        --ann-path   /data/musicavqa/json_update/avqa-test.json \
        --video-root /data/musicavqa/videos \
        --label      musicavqa_1k \
        --vllm --tp 8

── Comparison table only (no inference) ────────────────────────────────────────
    python eval_musicavqa.py \
        --compare \
            "fullb@1k:./eval_results/musicavqa/fullb@1k/metrics.json" \
            "fullb:./eval_results/musicavqa/fullb/metrics.json" \
            "musicavqa_1k:./eval_results/musicavqa/musicavqa_1k/metrics.json"

── Outputs (under --output-dir / --label) ──────────────────────────────────────
    eval_results.jsonl   — one JSON line per question (resumable)
    metrics.json         — aggregate + per-modality + per-subtype accuracy
    summary.txt          — human-readable summary (mirrors stdout)
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import io
import json
import os
import re
import shutil
import site
import tempfile
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_npp_lib = Path(site.getsitepackages()[0]) / "nvidia" / "npp" / "lib"
_npp_so  = _npp_lib / "libnppicc.so.12"
if _npp_so.is_file():
    ctypes.CDLL(str(_npp_so), mode=ctypes.RTLD_GLOBAL)

import torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path("./eval_results/musicavqa")

ANSWER_VOCAB: set[str] = {
    "yes", "no",
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "more than ten",
    "left", "right", "middle",
    "indoor", "outdoor",
    "simultaneously",
    "violin", "cello", "piano", "flute", "guitar", "acoustic_guitar",
    "electric_bass", "clarinet", "saxophone", "accordion", "trumpet",
    "tuba", "trombone", "horn", "ukulele", "banjo", "pipa", "guzheng",
    "erhu", "suona", "xylophone",
}

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

QUESTION_PROMPT = (
    "Watch the video carefully (audio enabled). "
    "Answer the question in one or a few words from the vocabulary. "
    "Question: {question}"
)

_PLACEHOLDER_RE = re.compile(r"<[A-Za-z][A-Za-z0-9_]*>")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Music-AVQA evaluation for Qwen3-Omni.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--base-model", type=str,
                   default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                   help="HF model id or local path.")
    p.add_argument("--adapter", type=str, default=None,
                   help="Path to DeepSpeed / HF LoRA adapter checkpoint.")
    # Data
    p.add_argument("--ann-path", type=Path, default=None,
                   help="avqa-test.json (MUSIC-AVQA json_update schema).")
    p.add_argument("--video-root", type=Path, default=None,
                   help="Dir containing {video_id}.mp4 files.")
    p.add_argument("--skip-deleted", action="store_true", default=True,
                   help="Drop records with question_deleted==1.")
    # Output
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Root output directory. Results go to <output-dir>/<label>/.")
    p.add_argument("--label", type=str, default=None,
                   help="Run label used for output subdir and summary header. "
                        "Defaults to adapter name or base-model name.")
    # Inference
    p.add_argument("--max-samples", type=int, default=-1,
                   help="Truncate evaluation set (-1 = all).")
    p.add_argument("--max-new-tokens", type=int, default=16,
                   help="Max tokens generated per answer.")
    p.add_argument("--temperature", type=float, default=0.0)
    # vLLM
    p.add_argument("--vllm", action="store_true", default=False,
                   help="Use vLLM for faster inference.")
    p.add_argument("--tp", type=int, default=None,
                   help="Tensor-parallel size for vLLM (default: all GPUs).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=32768)
    # Comparison-only mode
    p.add_argument("--compare", nargs="+", default=None,
                   metavar="LABEL:metrics.json",
                   help="Skip inference; load pre-computed metrics.json files "
                        "and print a comparison table. "
                        "Format: 'label:/path/to/metrics.json' (space-separated).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def render_question(question_content: str, templ_values_str: str) -> str:
    values = json.loads(templ_values_str)
    out = question_content
    for v in values:
        out = _PLACEHOLDER_RE.sub(str(v), out, count=1)
    return out


def parse_type(type_str: str) -> tuple[str, str]:
    parts = json.loads(type_str)
    return parts[0], parts[1]


def parse_answer(text: str) -> str:
    """Map free-text model output to the nearest vocab token (or '??')."""
    head = text.strip().lower()
    head = re.sub(r"[.,!?;:\"'()]", " ", head)
    for tok in sorted(ANSWER_VOCAB, key=lambda s: -len(s)):
        candidates = [tok, tok.replace("_", " ")] if "_" in tok else [tok]
        for c in candidates:
            if re.search(r"\b" + re.escape(c) + r"\b", head):
                return tok
    return "??"


def load_musicavqa(
    ann_path: Path,
    video_root: Path,
    skip_deleted: bool,
    max_samples: int,
) -> List[Dict[str, Any]]:
    with open(ann_path) as f:
        records = json.load(f)

    data, skipped = [], 0
    for rec in records:
        if skip_deleted and rec.get("question_deleted", 0) == 1:
            continue
        vid_path = video_root / f"{rec['video_id']}.mp4"
        if not vid_path.exists():
            skipped += 1
            continue
        try:
            question = render_question(rec["question_content"], rec["templ_values"])
            modality, subtype = parse_type(rec["type"])
        except (ValueError, KeyError):
            skipped += 1
            continue
        data.append({
            "question_id": str(rec.get("question_id", "")),
            "video_id":    rec["video_id"],
            "video_path":  str(vid_path),
            "question":    question,
            "gt_answer":   rec["anser"],
            "modality":    modality,
            "subtype":     subtype,
        })

    if skipped:
        print(f"[data] Skipped {skipped} records (missing video or bad schema)")
    if max_samples > 0:
        data = data[:max_samples]
    print(f"[data] {len(data)} questions ready for evaluation")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# HF inference
# ──────────────────────────────────────────────────────────────────────────────

def load_model_hf(base_model: str, adapter: Optional[str]):
    from omni_model_loading import load_qwen_omni_model
    model, processor, _ = load_qwen_omni_model(base_model, adapter)
    return model, processor


def run_inference_hf(
    model, processor,
    video_path: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    from qwen_omni_utils import process_mm_info

    tmp_dir = tempfile.mkdtemp(prefix="eval_mavqa_")
    symlink  = os.path.join(tmp_dir, "clip.mp4")
    os.symlink(os.path.abspath(video_path), symlink)

    prompt_text = QUESTION_PROMPT.format(question=question)
    conv = [{
        "role": "user",
        "content": [
            {"type": "video", "video": symlink},
            {"type": "text",  "text":  prompt_text},
        ],
    }]

    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )

    model_dtype = next(model.parameters()).dtype
    converted = {
        k: (v.to(model.device).to(model_dtype) if torch.is_floating_point(v) else v.to(model.device))
        if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    from omni_model_loading import is_omni_thinker_model
    is_thinker = is_omni_thinker_model(model)
    if is_thinker:
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
    else:
        gen_kwargs = {
            "thinker_max_new_tokens": max_new_tokens,
            "use_audio_in_video": True,
            "return_audio": False,
            "do_sample": temperature > 0,
        }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.inference_mode():
        output_ids = model.generate(**converted, **gen_kwargs)
    if isinstance(output_ids, tuple):
        output_ids = output_ids[0]

    prompt_len = converted["input_ids"].shape[1]
    response = processor.batch_decode(
        output_ids[:, prompt_len:], skip_special_tokens=True
    )[0].strip()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return response


# ──────────────────────────────────────────────────────────────────────────────
# vLLM inference
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_video_for_vllm(video_path: str):
    from qwen_omni_utils import process_mm_info
    import numpy as np

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": 2.0, "max_frames": 128},
            {"type": "text",  "text": "placeholder"},
        ],
    }]
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    video_np = (videos[0] * 255).byte().numpy()
    audio_tuple = None
    if audios:
        aud = audios[0]
        if isinstance(aud, tuple):
            audio_tuple = (
                aud[0].numpy() if hasattr(aud[0], "numpy") else aud[0], aud[1]
            )
        elif hasattr(aud, "numpy"):
            audio_tuple = (aud.numpy(), 16000)
    return video_np, audio_tuple


def build_vllm_prompt(question: str, base_model: str) -> str:
    from omni_model_loading import vllm_user_mm_prefix
    mm = vllm_user_mm_prefix(base_model, include_audio=True)
    prompt_text = QUESTION_PROMPT.format(question=question)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{mm}{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    n_correct  = sum(1 for r in results if r["pred_label"] == r["gt_answer"])
    non_match  = sum(1 for r in results if r["pred_label"] == "??")

    by_mod: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    by_sub: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        ok = int(r["pred_label"] == r["gt_answer"])
        by_mod[r.get("modality", "?")][0] += ok
        by_mod[r.get("modality", "?")][1] += 1
        by_sub[r.get("subtype",  "?")][0] += ok
        by_sub[r.get("subtype",  "?")][1] += 1

    def _fmt(d):
        return {
            m: {
                "accuracy":  round(corr / tot, 4) if tot else 0.0,
                "n_correct": corr,
                "n_total":   tot,
            }
            for m, (corr, tot) in sorted(d.items())
        }

    return {
        "total_samples":    total,
        "n_correct":        n_correct,
        "overall_accuracy": round(n_correct / total, 4),
        "non_match_rate":   round(non_match / total, 4),
        "per_modality":     _fmt(by_mod),
        "per_subtype":      _fmt(by_sub),
    }


def print_summary(metrics: Dict[str, Any], label: str) -> None:
    sep = "=" * 65
    print()
    print(sep)
    print(f"  Music-AVQA · {label}")
    print(sep)
    print(f"  Overall Accuracy : {metrics['overall_accuracy']:.1%}  "
          f"({metrics['n_correct']}/{metrics['total_samples']})")
    print(f"  Non-match rate   : {metrics['non_match_rate']:.1%}")
    print()
    print("  ─── Per Modality ───")
    for mod, d in metrics.get("per_modality", {}).items():
        print(f"    {mod:<22}: {d['accuracy']:.1%}  ({d['n_correct']}/{d['n_total']})")
    print()
    print("  ─── Per Sub-type ───")
    for sub, d in metrics.get("per_subtype", {}).items():
        print(f"    {sub:<25}: {d['accuracy']:.1%}  ({d['n_correct']}/{d['n_total']})")
    print(sep)


def print_comparison_table(entries: List[tuple[str, Dict[str, Any]]]) -> None:
    """Multi-run side-by-side comparison table."""
    if not entries:
        return
    all_mods = sorted({
        mod for _, m in entries for mod in m.get("per_modality", {})
    })
    lw, cw = 22, 13
    header = f"  {'Model':<{lw}}" + f"{'Overall':>{cw}}" + "".join(f"{m:>{cw}}" for m in all_mods)
    width = lw + cw * (1 + len(all_mods)) + 2
    sep = "=" * width

    print()
    print(sep)
    print("  Music-AVQA Comparison")
    print(sep)
    print(header)
    print("-" * width)
    for label, m in entries:
        oa = m.get("overall_accuracy", 0.0)
        row = f"  {label:<{lw}}{oa * 100:>{cw - 1}.1f}%"
        for mod in all_mods:
            md = m.get("per_modality", {}).get(mod)
            row += f"{md['accuracy'] * 100:>{cw - 1}.1f}%" if md else f"{'—':>{cw}}"
        print(row)
    print(sep)
    print()


def _save_and_finalize(
    results_jsonl: Path,
    metrics_json:  Path,
    summary_txt:   Path,
    args:          argparse.Namespace,
    label:         str,
) -> None:
    all_results = []
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                all_results.append(json.loads(line))

    if not all_results:
        print("[warn] No results to aggregate.")
        return

    metrics = compute_metrics(all_results)
    metrics["eval_config"] = {
        "base_model":    args.base_model,
        "adapter":       args.adapter,
        "ann_path":      str(args.ann_path),
        "video_root":    str(args.video_root),
        "max_new_tokens": args.max_new_tokens,
        "temperature":   args.temperature,
        "vllm":          args.vllm,
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print_summary(metrics, label)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_summary(metrics, label)
    summary_txt.write_text(buf.getvalue(), encoding="utf-8")

    print(f"\n[output] Results  : {results_jsonl}")
    print(f"[output] Metrics  : {metrics_json}")
    print(f"[output] Summary  : {summary_txt}")

    # Auto-compare with sibling runs already on disk
    siblings = []
    for sibling_dir in sorted(args.output_dir.iterdir()):
        sib_metrics = sibling_dir / "metrics.json"
        if sibling_dir.is_dir() and sib_metrics.exists() and sibling_dir.name != label:
            with open(sib_metrics) as f:
                siblings.append((sibling_dir.name, json.load(f)))
    if siblings:
        print_comparison_table(siblings + [(label, metrics)])


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Comparison-only mode ────────────────────────────────────────────────
    if args.compare is not None:
        entries = []
        for spec in args.compare:
            lbl, path = spec.split(":", 1)
            with open(path) as f:
                m = json.load(f)
            print_summary(m, lbl)
            entries.append((lbl, m))
        print_comparison_table(entries)
        return

    # ── Full eval mode ──────────────────────────────────────────────────────
    if args.ann_path is None or args.video_root is None:
        raise ValueError("--ann-path and --video-root are required for evaluation.")

    label = args.label or (
        Path(args.adapter).name if args.adapter else Path(args.base_model).name
    )
    out_dir      = args.output_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = out_dir / "eval_results.jsonl"
    metrics_json  = out_dir / "metrics.json"
    summary_txt   = out_dir / "summary.txt"

    test_data = load_musicavqa(
        args.ann_path, args.video_root, args.skip_deleted, args.max_samples
    )

    # Resume support
    processed: Dict[str, Dict] = {}
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                obj = json.loads(line)
                processed[obj["question_id"]] = obj
        print(f"[resume] {len(processed)} already done, skipping")

    todo = [item for item in test_data if item["question_id"] not in processed]

    # ── vLLM path ───────────────────────────────────────────────────────────
    if args.vllm:
        from vllm import LLM, SamplingParams

        tp = args.tp or torch.cuda.device_count()

        # Phase 1: CPU preprocess (GPUs idle)
        uniq_videos = list(dict.fromkeys(item["video_path"] for item in todo))
        print(
            f"[vllm] Phase 1 — CPU preprocess: {len(uniq_videos)} unique videos "
            f"for {len(todo)} samples (GPUs idle until model load).",
            flush=True,
        )
        preprocessed_v: Dict[str, Any] = {}
        preprocessed_a: Dict[str, Any] = {}
        failed_paths:   Set[str]       = set()

        for vp in tqdm(uniq_videos, desc="Preprocess video", unit="file"):
            try:
                vid_np, aud_tuple = preprocess_video_for_vllm(vp)
                preprocessed_v[vp] = vid_np
                if aud_tuple is not None:
                    preprocessed_a[vp] = aud_tuple
            except Exception as e:
                failed_paths.add(vp)
                print(f"  [skip] {Path(vp).name}: {e}")

        n_skip = sum(1 for item in todo if item["video_path"] in failed_paths)
        if failed_paths:
            print(f"[vllm] Preprocess failed for {len(failed_paths)} video(s), "
                  f"{n_skip} question(s) will be skipped.")

        # Phase 2: load model + run
        from omni_model_loading import cap_vllm_max_model_len
        vllm_max_len = cap_vllm_max_model_len(args.base_model, args.max_model_len)
        print(f"[vllm] Loading {args.base_model} tp={tp} max_model_len={vllm_max_len} …")
        llm = LLM(
            model=args.base_model,
            tensor_parallel_size=tp,
            max_model_len=vllm_max_len,
            max_num_seqs=4,
            limit_mm_per_prompt={"video": 1, "audio": 1},
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature if args.temperature > 0 else 0.0,
            top_p=0.9 if args.temperature > 0 else 1.0,
            max_tokens=args.max_new_tokens,
        )

        vllm_todo    = [i for i in todo if i["video_path"] not in failed_paths]
        fallback_todo = [i for i in todo if i["video_path"] in     failed_paths]
        print(f"[vllm] {len(vllm_todo)} samples ready, "
              f"{len(fallback_todo)} deferred to HF fallback …")

        with open(results_jsonl, "a", encoding="utf-8") as results_f:
            for idx, item in enumerate(tqdm(vllm_todo, desc="vLLM", unit="q")):
                if item["question_id"] in processed:
                    continue
                mm_data: Dict[str, Any] = {"video": preprocessed_v[item["video_path"]]}
                if item["video_path"] in preprocessed_a:
                    mm_data["audio"] = preprocessed_a[item["video_path"]]
                inp = {
                    "prompt":           build_vllm_prompt(item["question"], args.base_model),
                    "multi_modal_data": mm_data,
                }
                try:
                    outputs  = llm.generate([inp], sampling_params=sampling_params)
                    raw_text = outputs[0].outputs[0].text.strip()
                except (ValueError, RuntimeError) as exc:
                    print(f"  [error] {item['question_id']}: {exc}")
                    raw_text = ""

                pred_label = parse_answer(raw_text)
                record = {**item, "raw_output": raw_text, "pred_label": pred_label}
                processed[item["question_id"]] = record
                results_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_f.flush()

                if (idx + 1) % 200 == 0:
                    print(f"  [vllm] [{idx + 1}/{len(vllm_todo)}] done")

        preprocessed_v.clear()
        preprocessed_a.clear()

        if fallback_todo:
            print(f"[fallback] {len(fallback_todo)} samples → HF transformers …")
            del llm
            gc.collect()
            torch.cuda.empty_cache()

            model, processor = load_model_hf(args.base_model, args.adapter)
            with open(results_jsonl, "a", encoding="utf-8") as results_f:
                for item in tqdm(fallback_todo, desc="Fallback", unit="q"):
                    if item["question_id"] in processed:
                        continue
                    try:
                        raw_text = run_inference_hf(
                            model, processor,
                            item["video_path"], item["question"],
                            args.max_new_tokens, args.temperature,
                        )
                    except Exception as exc:
                        import traceback
                        print(f"  [error] {item['question_id']}: {exc}")
                        traceback.print_exc()
                        raw_text = ""
                    pred_label = parse_answer(raw_text)
                    record = {**item, "raw_output": raw_text, "pred_label": pred_label}
                    processed[item["question_id"]] = record
                    results_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    results_f.flush()
                    gc.collect()
                    torch.cuda.empty_cache()

    # ── HF path ──────────────────────────────────────────────────────────────
    else:
        print(f"[hf] Loading {args.base_model} …")
        model, processor = load_model_hf(args.base_model, args.adapter)

        with open(results_jsonl, "a", encoding="utf-8") as results_f:
            for item in tqdm(todo, desc="Music-AVQA", unit="q"):
                try:
                    raw_text = run_inference_hf(
                        model, processor,
                        item["video_path"], item["question"],
                        args.max_new_tokens, args.temperature,
                    )
                except Exception as exc:
                    import traceback
                    print(f"  [error] {item['question_id']}: {exc}")
                    traceback.print_exc()
                    raw_text = ""
                pred_label = parse_answer(raw_text)
                record = {**item, "raw_output": raw_text, "pred_label": pred_label}
                processed[item["question_id"]] = record
                results_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_f.flush()
                gc.collect()
                torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Aggregate + save ─────────────────────────────────────────────────────
    _save_and_finalize(results_jsonl, metrics_json, summary_txt, args, label)


if __name__ == "__main__":
    main()
