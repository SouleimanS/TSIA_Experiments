"""Print a comparison table of AVHBench eval results across variants.

Scans `results/avhbench_*.json` and produces a markdown table comparing
per-task and overall accuracy. Optionally compares against paper
baselines for context.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --csv         # also write results_table.csv
    python scripts/compare_results.py --no-paper    # skip paper baselines
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

RESULTS_DIR = Path("results")

# Paper Table 1 numbers (AVHBench paper, ICLR 2025) for context.
# These ran on the FULL 5,302-item test set, so they are directly comparable
# to our AVQA-trained variants (also evaluated on full set).
# Combined-trained variants are evaluated on the 1,060-item held-out split,
# so the paper rows are NOT comparable to combined runs.
PAPER_BASELINES = {
    "PandaGPT":              {"a2v": 0.585, "v2a": 0.613, "avm": 0.512, "overall": 0.570},
    "Video-LLaMA":           {"a2v": 0.501, "v2a": 0.502, "avm": 0.500, "overall": 0.501},
    "ImageBind-LLM":         {"a2v": 0.503, "v2a": 0.500, "avm": 0.500, "overall": 0.501},
    "ChatBridge":            {"a2v": 0.529, "v2a": 0.328, "avm": 0.299, "overall": 0.385},
    "OneLLM":                {"a2v": 0.537, "v2a": 0.443, "avm": 0.601, "overall": 0.527},
    "Video-SALMONN":         {"a2v": 0.781, "v2a": 0.652, "avm": None,  "overall": None},
    "Video-LLaMA2":          {"a2v": 0.752, "v2a": 0.742, "avm": None,  "overall": None},
    "Gemini-Flash":          {"a2v": 0.833, "v2a": 0.630, "avm": None,  "overall": None},
    "AVHModel-Align-FT":     {"a2v": 0.839, "v2a": 0.773, "avm": 0.556, "overall": None},
}

TASK_KEY_MAP = {
    "Audio-driven Video Hallucination": "a2v",
    "Video-driven Audio Hallucination": "v2a",
    "AV Matching": "avm",
}


def load_result(path):
    with open(path) as f:
        data = json.load(f)
    row = {}
    for task_name, short in TASK_KEY_MAP.items():
        m = data.get("per_task", {}).get(task_name)
        row[short] = m["acc"] if m else None
    row["overall"] = data.get("overall", {}).get("acc")
    row["n_items"] = data.get("n_items", 0)
    return row


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--no-paper", action="store_true")
    args = ap.parse_args()

    files = sorted(RESULTS_DIR.glob("avhbench_*.json"))
    if not files:
        print(f"No results files found in {RESULTS_DIR}")
        return

    rows = []
    for f in files:
        variant = f.stem.replace("avhbench_", "")
        r = load_result(f)
        rows.append({"variant": variant, **r, "source": "ours"})

    print()
    print("=" * 80)
    print("AVHBench evaluation comparison")
    print("=" * 80)
    print()
    print(f"{'Variant':<28} {'A->V':>7} {'V->A':>7} {'AVMat':>7} {'Overall':>9} {'N':>7}")
    print(f"{'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*7}")

    if not args.no_paper:
        print("# Paper baselines (Table 1, full 5302-item test):")
        for name, m in PAPER_BASELINES.items():
            print(f"{name:<28} {fmt(m['a2v']):>7} {fmt(m['v2a']):>7} {fmt(m['avm']):>7} {fmt(m['overall']):>9} {'5302':>7}")
        print()

    print("# Our variants:")
    for r in rows:
        print(f"{r['variant']:<28} {fmt(r['a2v']):>7} {fmt(r['v2a']):>7} {fmt(r['avm']):>7} {fmt(r['overall']):>9} {r['n_items']:>7}")

    print()
    print("=" * 80)
    print("Markdown table (copy/paste)")
    print("=" * 80)
    print()
    print("| Variant | A->V | V->A | AVMat | Overall | N |")
    print("|---|---|---|---|---|---|")
    if not args.no_paper:
        for name, m in PAPER_BASELINES.items():
            print(f"| _{name}_ | {fmt(m['a2v'])} | {fmt(m['v2a'])} | {fmt(m['avm'])} | {fmt(m['overall'])} | 5302 |")
    for r in rows:
        print(f"| **{r['variant']}** | {fmt(r['a2v'])} | {fmt(r['v2a'])} | {fmt(r['avm'])} | {fmt(r['overall'])} | {r['n_items']} |")

    if args.csv:
        out = RESULTS_DIR / "comparison_table.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "a2v", "v2a", "avm", "overall", "n_items", "source"])
            if not args.no_paper:
                for name, m in PAPER_BASELINES.items():
                    w.writerow([name, m["a2v"], m["v2a"], m["avm"], m["overall"], 5302, "paper"])
            for r in rows:
                w.writerow([r["variant"], r["a2v"], r["v2a"], r["avm"], r["overall"], r["n_items"], "ours"])
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()