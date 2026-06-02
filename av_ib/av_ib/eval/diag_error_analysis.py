#!/usr/bin/env python3
"""Diagnostic 1 — Error analysis from eval result files.

Consumes the per-record eval outputs (CSV from v6_eval.py, or eval_results.jsonl
from eval_musicavqa.py) for any set of models and produces:

  - Top-line accuracy + 95% Wilson CI
  - Accuracy by MODALITY  (Audio / Visual / Audio-Visual)
  - Accuracy by QUESTION CLASS (Counting / Existential / Comparative / Location / Temporal)
  - Prediction-distribution collapse detection (is the model just saying "yes"?)
  - Yes/No calibration (predicted-yes rate vs gold-yes rate)
  - Pairwise disagreement matrix between models
  - Per-(modality x class) cell table, the single most informative view

Usage:
  python diag_error_analysis.py \
      --result baseline:runs/baseline_reeval_1000.csv \
      --result b@1k:runs/v6_tier1_b/eval_step1000_1000.csv \
      --result bbis@1k:results/eval/musicavqa/musicavqa_bis_1k/eval_results.jsonl \
      --out-dir results/diagnostics

Accepts both .csv (with columns video_id, prompt/question, type, correct,
parsed_pred/pred_label, gold/gt_answer) and .jsonl (one dict per line).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# IO: normalize CSV and JSONL into a common record schema
# ──────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return " ".join(str(s).split()).rstrip("?").strip().lower()


def _parse_type(raw: Any) -> Tuple[str, str]:
    """Return (modality, qclass). type may be a JSON list string or a python list."""
    if isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        try:
            parts = ast.literal_eval(raw)
        except Exception:
            try:
                parts = json.loads(raw)
            except Exception:
                parts = [str(raw)]
    if not isinstance(parts, (list, tuple)):
        parts = [str(parts)]
    modality = str(parts[0]) if len(parts) > 0 else "?"
    qclass = str(parts[1]) if len(parts) > 1 else "?"
    return modality, qclass


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load CSV or JSONL into a list of normalized record dicts."""
    records = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                modality = obj.get("modality")
                qclass = obj.get("subtype")
                if modality is None and "type" in obj:
                    modality, qclass = _parse_type(obj["type"])
                records.append({
                    "video_id": obj.get("video_id", ""),
                    "question": _norm(obj.get("question", obj.get("prompt", ""))),
                    "modality": modality or "?",
                    "qclass": qclass or "?",
                    "pred": str(obj.get("pred_label", obj.get("parsed_pred", "??"))).lower(),
                    "gold": str(obj.get("gt_answer", obj.get("gold", obj.get("answer", "")))).lower(),
                    "correct": (str(obj.get("pred_label", obj.get("parsed_pred", "??"))).lower()
                                == str(obj.get("gt_answer", obj.get("gold", ""))).lower())
                               if "correct" not in obj else _truthy(obj["correct"]),
                })
    else:  # CSV
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                modality, qclass = _parse_type(row.get("type", "[]"))
                pred = str(row.get("parsed_pred", row.get("pred_label", "??"))).lower()
                gold = str(row.get("gold", row.get("gt_answer", row.get("answer", "")))).lower()
                if "correct" in row:
                    correct = _truthy(row["correct"])
                else:
                    correct = (pred == gold)
                records.append({
                    "video_id": row.get("video_id", ""),
                    "question": _norm(row.get("prompt", row.get("question", ""))),
                    "modality": modality,
                    "qclass": qclass,
                    "pred": pred,
                    "gold": gold,
                    "correct": correct,
                })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def acc(records: List[Dict]) -> Tuple[float, int, int]:
    n = len(records)
    k = sum(1 for r in records if r["correct"])
    return (k / n if n else 0.0, k, n)


# ──────────────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────────────

def report_topline(models: Dict[str, List[Dict]]) -> Dict:
    print("=" * 78)
    print("TOP-LINE ACCURACY (95% Wilson CI)")
    print("=" * 78)
    print(f"{'model':<22}{'n':>6}{'correct':>9}{'acc':>9}{'95% CI':>20}{'unparsed':>10}")
    out = {}
    for name, recs in models.items():
        a, k, n = acc(recs)
        lo, hi = wilson_ci(k, n)
        unparsed = sum(1 for r in recs if r["pred"] == "??")
        print(f"{name:<22}{n:>6}{k:>9}{a:>8.1%}"
              f"{f'[{lo:.1%}, {hi:.1%}]':>20}{unparsed:>10}")
        out[name] = {"acc": a, "k": k, "n": n, "ci": [lo, hi], "unparsed": unparsed}
    print()
    return out


def report_by_field(models: Dict[str, List[Dict]], field: str, title: str) -> Dict:
    print("=" * 78)
    print(f"ACCURACY BY {title}")
    print("=" * 78)
    cats = sorted({r[field] for recs in models.values() for r in recs})
    header = f"{field:<16}" + "".join(f"{name[:13]:>15}" for name in models)
    print(header)
    print("-" * len(header))
    out = defaultdict(dict)
    for cat in cats:
        line = f"{cat:<16}"
        for name, recs in models.items():
            subset = [r for r in recs if r[field] == cat]
            if subset:
                a, k, n = acc(subset)
                line += f"{f'{a:.0%} ({n})':>15}"
                out[cat][name] = {"acc": a, "k": k, "n": n}
            else:
                line += f"{'—':>15}"
        print(line)
    print()
    return dict(out)


def report_collapse(models: Dict[str, List[Dict]]) -> Dict:
    print("=" * 78)
    print("PREDICTION DISTRIBUTION (collapse detection — top 6 predictions)")
    print("=" * 78)
    out = {}
    for name, recs in models.items():
        c = Counter(r["pred"] for r in recs)
        n = len(recs)
        print(f"\n{name}:")
        top = c.most_common(6)
        out[name] = {"top": [(v, cnt, cnt / n) for v, cnt in top],
                     "distinct": len(c)}
        for v, cnt in top:
            bar = "█" * int(40 * cnt / n)
            print(f"  {v!r:<16}{cnt:>5} ({cnt/n:>5.1%}) {bar}")
        print(f"  distinct predictions: {len(c)}")
    print()
    return out


def report_yesno_calibration(models: Dict[str, List[Dict]]) -> Dict:
    print("=" * 78)
    print("YES/NO CALIBRATION (pred-yes rate vs gold-yes rate)")
    print("=" * 78)
    print(f"{'model':<22}{'n_yn':>6}{'acc':>8}{'pred_yes':>10}{'gold_yes':>10}{'bias':>8}")
    out = {}
    for name, recs in models.items():
        yn = [r for r in recs if r["gold"] in ("yes", "no")]
        if not yn:
            continue
        n = len(yn)
        a, k, _ = acc(yn)
        pred_yes = sum(1 for r in yn if r["pred"] == "yes") / n
        gold_yes = sum(1 for r in yn if r["gold"] == "yes") / n
        bias = pred_yes - gold_yes
        print(f"{name:<22}{n:>6}{a:>7.1%}{pred_yes:>10.1%}{gold_yes:>10.1%}{bias:>+8.1%}")
        out[name] = {"n": n, "acc": a, "pred_yes": pred_yes,
                     "gold_yes": gold_yes, "bias": bias}
    print()
    return out


def report_disagreement(models: Dict[str, List[Dict]]) -> Dict:
    """Pairwise agreement, keyed on (video_id, question)."""
    names = list(models)
    if len(names) < 2:
        return {}
    print("=" * 78)
    print("PAIRWISE DISAGREEMENT (on shared records)")
    print("=" * 78)
    # Build key→correct maps
    keyed = {}
    for name, recs in models.items():
        keyed[name] = {(r["video_id"], r["question"]): r["correct"] for r in recs}
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = set(keyed[a]) & set(keyed[b])
            n = len(shared)
            if n == 0:
                continue
            both_c = sum(1 for k in shared if keyed[a][k] and keyed[b][k])
            both_w = sum(1 for k in shared if not keyed[a][k] and not keyed[b][k])
            a_only = sum(1 for k in shared if keyed[a][k] and not keyed[b][k])
            b_only = sum(1 for k in shared if not keyed[a][k] and keyed[b][k])
            disagree = (a_only + b_only) / n
            print(f"\n{a}  vs  {b}   (shared n={n})")
            print(f"  both correct : {both_c/n:>6.1%}")
            print(f"  both wrong   : {both_w/n:>6.1%}")
            print(f"  {a[:12]:<12} only : {a_only/n:>6.1%}")
            print(f"  {b[:12]:<12} only : {b_only/n:>6.1%}")
            print(f"  >> disagreement: {disagree:.1%}")
            out[f"{a}_vs_{b}"] = {
                "n": n, "both_correct": both_c / n, "both_wrong": both_w / n,
                f"{a}_only": a_only / n, f"{b}_only": b_only / n,
                "disagreement": disagree,
            }
    print()
    return out


def report_cell_table(models: Dict[str, List[Dict]]) -> Dict:
    """The most informative view: accuracy per (modality × qclass) cell."""
    print("=" * 78)
    print("ACCURACY PER (MODALITY × QUESTION CLASS) CELL")
    print("=" * 78)
    out = {}
    for name, recs in models.items():
        cells = defaultdict(lambda: [0, 0])
        for r in recs:
            cell = (r["modality"], r["qclass"])
            cells[cell][0] += int(r["correct"])
            cells[cell][1] += 1
        print(f"\n{name}:")
        print(f"  {'modality':<14}{'class':<14}{'n':>5}{'acc':>8}")
        out[name] = {}
        for (mod, cls), (k, n) in sorted(cells.items(), key=lambda x: -x[1][1]):
            print(f"  {mod:<14}{cls:<14}{n:>5}{k/n:>8.1%}")
            out[name][f"{mod}|{cls}"] = {"acc": k / n, "k": k, "n": n}
    print()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result", action="append", required=True,
                   metavar="LABEL:PATH",
                   help="label:path pair. Repeat for each model. "
                        "Path is .csv or .jsonl per-record eval output.")
    p.add_argument("--out-dir", default="results/diagnostics")
    args = p.parse_args()

    models: Dict[str, List[Dict]] = {}
    for spec in args.result:
        label, path = spec.split(":", 1)
        recs = load_records(Path(path))
        models[label] = recs
        print(f"[load] {label}: {len(recs)} records from {path}")
    print()

    summary = {
        "topline":        report_topline(models),
        "by_modality":    report_by_field(models, "modality", "MODALITY"),
        "by_qclass":      report_by_field(models, "qclass", "QUESTION CLASS"),
        "collapse":       report_collapse(models),
        "yesno_calib":    report_yesno_calibration(models),
        "disagreement":   report_disagreement(models),
        "cell_table":     report_cell_table(models),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "error_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out_dir / 'error_analysis.json'}")


if __name__ == "__main__":
    main()
