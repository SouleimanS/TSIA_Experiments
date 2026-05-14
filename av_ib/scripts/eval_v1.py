"""Eval v1 on AVHBench-test. Smoke test by default (50 records).

Usage:
    python scripts/eval_v1.py                    # 50 records, no checkpoint
    python scripts/eval_v1.py --n 6408            # full eval
    python scripts/eval_v1.py --ckpt path/to.pt   # load trainable state
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from av_ib.model.av_model_v1 import AVModelV1
from av_ib.eval.avhbench import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50,
                        help="Number of records to evaluate (default 50, full=6408)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to trainable_state checkpoint (.pt)")
    parser.add_argument("--qa", type=str,
                        default=str(Path.home() / "SOULEIMAN_repo" / "datasets" /
                                    "AVHBench" / "data" / "AVHBench_v0" / "json" / "qa.json"))
    parser.add_argument("--video-dir", type=str,
                        default=str(Path.home() / "SOULEIMAN_repo" / "datasets" /
                                    "AVHBench" / "data" / "AVHBench_v0" / "video"))
    parser.add_argument("--output", type=str, default="predictions_v1.csv")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    print("=== Building AVModelV1 ===")
    model = AVModelV1(use_lora=True).cuda()

    if args.ckpt is not None:
        print(f"=== Loading checkpoint {args.ckpt} ===")
        ckpt = torch.load(args.ckpt, map_location="cuda")
        state = ckpt["trainable_state"] if "trainable_state" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    print(f"\n=== Running eval on {args.n} records ===\n")
    run_eval(
        model,
        qa_json_path=args.qa,
        video_dir=args.video_dir,
        output_csv=args.output,
        max_records=args.n,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
