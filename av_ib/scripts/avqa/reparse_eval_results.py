#!/usr/bin/env python3
"""Re-parse eval results with the correct AVQA answer parser.

The original eval used the MUSIC-AVQA vocab parser which doesn't recognize
A/B/C/D letters, producing 0% accuracy for all models. This script reads
the raw predictions from results.jsonl / *_riskcoverage.json and recomputes
correct metrics.

Usage:
    python scripts/avqa/reparse_eval_results.py --eval-dir runs/avqa/eval
"""
import argparse
import json
import re
from pathlib import Path


def parse_letter(raw: str) -> str:
    """Extract first A/B/C/D letter from model output."""
    if not raw or str(raw).startswith("<ERROR"):
        return "??"
    m = re.search(r"\b([A-Da-d])\b", str(raw))
    return m.group(1).lower() if m else "??"


def reparse_dir(model_dir: Path) -> dict:
    jsonl = model_dir / "results.jsonl"
    if not jsonl.exists():
        print(f"  SKIP {model_dir.name} — no results.jsonl")
        return {}

    records = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]

    n_correct = 0
    by_relation = {}
    nlls, kl_vs, kl_as = [], [], []

    for r in records:
        # actual field names: raw_pred, gold, nll, kl_v, kl_a, modality
        raw  = r.get("raw_pred", r.get("raw", ""))
        gold = r.get("gold", "").strip().lower()
        parsed  = parse_letter(raw)
        correct = int(parsed == gold)
        n_correct += correct

        rel = r.get("modality") or r.get("question_relation") or "?"
        if rel not in by_relation:
            by_relation[rel] = [0, 0]
        by_relation[rel][0] += correct
        by_relation[rel][1] += 1

        if r.get("nll") is not None:
            nlls.append(r["nll"])
        if r.get("kl_v") is not None:
            kl_vs.append(r["kl_v"])
        if r.get("kl_a") is not None:
            kl_as.append(r["kl_a"])

    n = len(records)
    acc = n_correct / max(n, 1)

    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 1),
        "mean_nll": round(sum(nlls) / len(nlls), 4) if nlls else None,
        "mean_kl_v": round(sum(kl_vs) / len(kl_vs), 2) if kl_vs else None,
        "mean_kl_a": round(sum(kl_as) / len(kl_as), 2) if kl_as else None,
        "per_relation": {
            k: {"acc_pct": round(100 * v[0] / max(v[1], 1), 1), "n": v[1], "n_correct": v[0]}
            for k, v in sorted(by_relation.items())
        },
    }


def reparse_riskcoverage(rc_json: Path) -> dict:
    data = json.loads(rc_json.read_text())
    per_sample = data.get("per_sample", [])
    if not per_sample:
        return data

    by_corr = {}
    for r in per_sample:
        corr      = r.get("corruption", "?")
        raw_pred  = r.get("raw_pred", r.get("pred", ""))
        gold      = r.get("gold", "").strip().lower()
        abstained = r.get("abstained", False)

        if corr not in by_corr:
            by_corr[corr] = {"n": 0, "n_abstained": 0, "n_answered": 0, "n_correct": 0}
        s = by_corr[corr]
        s["n"] += 1
        if abstained:
            s["n_abstained"] += 1
        else:
            s["n_answered"] += 1
            if parse_letter(raw_pred) == gold:
                s["n_correct"] += 1

    summary = {}
    for corr, s in by_corr.items():
        cov  = s["n_answered"] / max(s["n"], 1)
        risk = 1.0 - s["n_correct"] / max(s["n_answered"], 1)
        summary[corr] = {
            "coverage_pct": round(cov * 100, 1),
            "risk_pct": round(risk * 100, 1),
            "n": s["n"],
            "n_abstained": s["n_abstained"],
            "n_answered": s["n_answered"],
            "n_correct": s["n_correct"],
        }
    data["summary_reparsed"] = summary
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="runs/avqa/eval")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else None

    print("=" * 60)
    print("REPARSED ACCURACY (letter A-D matching)")
    print("=" * 60)

    for model_dir in sorted(eval_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        metrics = reparse_dir(model_dir)
        if not metrics:
            continue

        dst = (out_dir / model_dir.name / "metrics_reparsed.json"
               if out_dir else model_dir / "metrics_reparsed.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(metrics, indent=2))

        print(f"\n  {model_dir.name}")
        print(f"    accuracy : {metrics['accuracy_pct']:.1f}%  ({metrics['n_correct']}/{metrics['n']})")
        print(f"    mean_nll : {metrics['mean_nll']}  kl_v: {metrics['mean_kl_v']}")
        for k, v in metrics["per_relation"].items():
            print(f"    [{k:>5}]  {v['acc_pct']:5.1f}%  (n={v['n']})")

    print("\n" + "=" * 60)
    print("REPARSED RISK-COVERAGE")
    print("=" * 60)

    for rc_file in sorted(eval_dir.glob("*_riskcoverage.json")):
        data = reparse_riskcoverage(rc_file)
        dst = (out_dir / rc_file.name if out_dir
               else rc_file.parent / (rc_file.stem + "_reparsed.json"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(data, indent=2))

        print(f"\n  {rc_file.stem}")
        smry = data.get("summary_reparsed", {})
        if not smry:
            print("    (no per_sample data)")
            continue
        print(f"  {'corruption':<20} {'cov':>6} {'risk':>6}  {'correct/answered':>16}")
        for corr, v in smry.items():
            print(f"  {corr:<20} {v['coverage_pct']:5.1f}% {v['risk_pct']:5.1f}%"
                  f"  {v['n_correct']:>4}/{v['n_answered']:<4}")

    print("\ndone.")


if __name__ == "__main__":
    main()
