#!/usr/bin/env python3
"""Diagnostic 2 — Data distribution analysis.

Answers: "what does the data itself make easy or hard, and what is the
majority-class ceiling the model must beat to show it's actually learning?"

Produces, for the MUSIC-AVQA annotation file(s):

  - Record counts: total, deleted, video-on-disk, real vs synthetic
  - Per-(modality × class) record counts
  - Answer-vocabulary distribution + Shannon entropy per class
  - MAJORITY-CLASS baseline accuracy per class (the number to beat)
  - BLIND priors: "always yes", "always most-common-answer" accuracy
  - Question-length and template-placeholder statistics
  - Train/test answer-distribution drift (if both files given)

Usage:
  python diag_data_analysis.py \
      --train /path/to/avqa-train.json \
      --test  /path/to/avqa-test.json \
      --video-root /path/to/videos/raw/MUCIS-AVQA-videos-Synthetic \
      --out-dir results/diagnostics
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_PLACEHOLDER_RE = re.compile(r"<[A-Za-z][A-Za-z0-9_]*>")


def render_question(qc: str, templ: str) -> str:
    try:
        values = json.loads(templ)
    except Exception:
        return qc
    out = qc
    for v in values:
        out = _PLACEHOLDER_RE.sub(str(v), out, count=1)
    return out


def parse_type(raw: Any) -> Tuple[str, str]:
    try:
        parts = json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        parts = [str(raw)]
    return (str(parts[0]) if parts else "?",
            str(parts[1]) if len(parts) > 1 else "?")


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def load(path: Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────

def section_counts(records: List[Dict], video_root: Optional[Path], tag: str) -> Dict:
    print("=" * 78)
    print(f"RECORD COUNTS — {tag}")
    print("=" * 78)
    total = len(records)
    deleted = sum(1 for r in records if r.get("question_deleted", 0) == 1)
    live = [r for r in records if r.get("question_deleted", 0) == 0]
    real = sum(1 for r in live if str(r["video_id"]).startswith("00"))
    synth = len(live) - real

    on_disk = None
    if video_root:
        on_disk = sum(1 for r in live
                      if (video_root / f"{r['video_id']}.mp4").exists())

    print(f"  total records        : {total}")
    print(f"  deleted              : {deleted}")
    print(f"  live                 : {len(live)}")
    print(f"  real videos (00*)    : {real}")
    print(f"  synthetic videos     : {synth}")
    if on_disk is not None:
        print(f"  video file on disk   : {on_disk}  "
              f"({on_disk/len(live):.1%} of live)")
        print(f"  >> UNAVAILABLE       : {len(live)-on_disk}  "
              f"(cannot train/eval on these)")
    print()
    return {"total": total, "deleted": deleted, "live": len(live),
            "real": real, "synthetic": synth, "on_disk": on_disk}


def section_cells(records: List[Dict], tag: str) -> Dict:
    print("=" * 78)
    print(f"RECORDS PER (MODALITY × CLASS) — {tag}")
    print("=" * 78)
    cells = Counter()
    for r in records:
        if r.get("question_deleted", 0) == 1:
            continue
        mod, cls = parse_type(r["type"])
        cells[(mod, cls)] += 1
    print(f"  {'modality':<14}{'class':<16}{'n':>6}")
    out = {}
    for (mod, cls), n in sorted(cells.items(), key=lambda x: -x[1]):
        print(f"  {mod:<14}{cls:<16}{n:>6}")
        out[f"{mod}|{cls}"] = n
    print()
    return out


def section_answer_priors(records: List[Dict], tag: str) -> Dict:
    """Per-class: answer entropy + majority-class accuracy ceiling."""
    print("=" * 78)
    print(f"ANSWER DISTRIBUTION & MAJORITY-CLASS CEILING — {tag}")
    print("=" * 78)
    by_class: Dict[str, Counter] = defaultdict(Counter)
    overall = Counter()
    for r in records:
        if r.get("question_deleted", 0) == 1:
            continue
        _, cls = parse_type(r["type"])
        ans = str(r["anser"]).lower()
        by_class[cls][ans] += 1
        overall[ans] += 1

    print(f"  {'class':<16}{'n':>6}{'distinct':>10}{'entropy':>9}"
          f"{'majority':>22}{'maj_acc':>9}")
    out = {}
    for cls, counts in sorted(by_class.items(), key=lambda x: -sum(x[1].values())):
        n = sum(counts.values())
        ent = entropy(counts)
        maj_ans, maj_n = counts.most_common(1)[0]
        maj_acc = maj_n / n
        print(f"  {cls:<16}{n:>6}{len(counts):>10}{ent:>9.2f}"
              f"{maj_ans[:18]:>22}{maj_acc:>9.1%}")
        out[cls] = {"n": n, "distinct": len(counts), "entropy": ent,
                    "majority_answer": maj_ans, "majority_acc": maj_acc}

    # Blind whole-dataset priors
    n_all = sum(overall.values())
    maj_all, maj_all_n = overall.most_common(1)[0]
    print()
    print(f"  BLIND PRIORS (whole {tag}):")
    print(f"    always '{maj_all}'   → {maj_all_n/n_all:.1%}")
    yes_n = overall.get("yes", 0)
    yn_total = overall.get("yes", 0) + overall.get("no", 0)
    if yn_total:
        print(f"    yes/no subset: always 'yes' → "
              f"{yes_n/yn_total:.1%}  (of {yn_total} y/n records)")
    print(f"    overall answer entropy: {entropy(overall):.2f} bits "
          f"({len(overall)} distinct answers)")
    print()
    out["_blind"] = {
        "majority_answer": maj_all,
        "majority_acc": maj_all_n / n_all,
        "overall_entropy": entropy(overall),
        "distinct_answers": len(overall),
    }
    return out


def section_question_stats(records: List[Dict], tag: str) -> Dict:
    print("=" * 78)
    print(f"QUESTION-TEXT STATISTICS — {tag}")
    print("=" * 78)
    lengths = []
    placeholder_count = 0
    for r in records:
        if r.get("question_deleted", 0) == 1:
            continue
        q = render_question(r["question_content"], r["templ_values"])
        lengths.append(len(q.split()))
        if _PLACEHOLDER_RE.search(r["question_content"]):
            placeholder_count += 1
    lengths.sort()
    n = len(lengths)
    mean_len = sum(lengths) / n if n else 0
    median_len = lengths[n // 2] if n else 0
    print(f"  questions            : {n}")
    print(f"  mean length (words)  : {mean_len:.1f}")
    print(f"  median length        : {median_len}")
    print(f"  with template slots  : {placeholder_count} ({placeholder_count/n:.1%})")
    print()
    return {"n": n, "mean_len": mean_len, "median_len": median_len,
            "with_placeholders": placeholder_count}


def section_drift(train: List[Dict], test: List[Dict]) -> Dict:
    """Answer-distribution drift between train and test, per class."""
    print("=" * 78)
    print("TRAIN→TEST ANSWER DISTRIBUTION DRIFT (total variation distance)")
    print("=" * 78)

    def class_answer_dist(recs):
        d = defaultdict(Counter)
        for r in recs:
            if r.get("question_deleted", 0) == 1:
                continue
            _, cls = parse_type(r["type"])
            d[cls][str(r["anser"]).lower()] += 1
        return d

    dtr = class_answer_dist(train)
    dte = class_answer_dist(test)
    print(f"  {'class':<16}{'TV distance':>14}")
    out = {}
    for cls in sorted(set(dtr) | set(dte)):
        tr, te = dtr.get(cls, Counter()), dte.get(cls, Counter())
        ntr, nte = sum(tr.values()), sum(te.values())
        if ntr == 0 or nte == 0:
            continue
        vocab = set(tr) | set(te)
        tv = 0.5 * sum(abs(tr[a] / ntr - te[a] / nte) for a in vocab)
        flag = "  <-- high drift" if tv > 0.15 else ""
        print(f"  {cls:<16}{tv:>14.3f}{flag}")
        out[cls] = tv
    print()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, default=None)
    p.add_argument("--video-root", type=Path, default=None)
    p.add_argument("--out-dir", default="results/diagnostics")
    args = p.parse_args()

    train = load(args.train)
    summary = {"train": {}}

    summary["train"]["counts"] = section_counts(train, args.video_root, "TRAIN")
    summary["train"]["cells"] = section_cells(train, "TRAIN")
    summary["train"]["priors"] = section_answer_priors(train, "TRAIN")
    summary["train"]["questions"] = section_question_stats(train, "TRAIN")

    if args.test:
        test = load(args.test)
        summary["test"] = {}
        summary["test"]["counts"] = section_counts(test, args.video_root, "TEST")
        summary["test"]["cells"] = section_cells(test, "TEST")
        summary["test"]["priors"] = section_answer_priors(test, "TEST")
        summary["test"]["questions"] = section_question_stats(test, "TEST")
        summary["drift"] = section_drift(train, test)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out_dir / 'data_analysis.json'}")


if __name__ == "__main__":
    main()
