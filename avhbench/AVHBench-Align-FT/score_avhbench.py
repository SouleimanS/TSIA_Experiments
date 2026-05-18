"""Score AVHBench predictions."""
import argparse
import csv
import re
from collections import defaultdict


def extract_yes_no(text):
    if not text:
        return None
    head = text[:50].lower()
    m_yes = re.search(r"\byes\b", head)
    m_no = re.search(r"\bno\b", head)
    if m_yes and m_no:
        return "Yes" if m_yes.start() < m_no.start() else "No"
    if m_yes:
        return "Yes"
    if m_no:
        return "No"
    return None


def compute_binary_metrics(records, pos_label="Yes"):
    n_total = len(records)
    parseable = [r for r in records if r["pred"] is not None]
    n_parse = len(parseable)
    n_unparse = n_total - n_parse
    tp = sum(1 for r in parseable if r["gold"] == pos_label and r["pred"] == pos_label)
    tn = sum(1 for r in parseable if r["gold"] != pos_label and r["pred"] != pos_label)
    fp = sum(1 for r in parseable if r["gold"] != pos_label and r["pred"] == pos_label)
    fn = sum(1 for r in parseable if r["gold"] == pos_label and r["pred"] != pos_label)
    acc = (tp + tn) / n_parse if n_parse else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    yes_rate = (tp + fp) / n_parse if n_parse else 0.0
    return {
        "n_total": n_total, "n_parseable": n_parse, "n_unparseable": n_unparse,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "yes_rate": yes_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()

    by_task = defaultdict(list)
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["task"]].append(row)

    print(f"\nLoaded {sum(len(v) for v in by_task.values())} records across {len(by_task)} tasks\n")

    judgment_tasks = [
        "Video-driven Audio Hallucination",
        "Audio-driven Video Hallucination",
        "AV Matching",
    ]
    for task in judgment_tasks:
        if task not in by_task:
            continue
        records = [
            {"gold": r["gold_label"], "pred": extract_yes_no(r["prediction"])}
            for r in by_task[task]
        ]
        m = compute_binary_metrics(records)
        print(f"=== {task} ===")
        print(f"  n={m['n_total']}  parseable={m['n_parseable']}  unparseable={m['n_unparseable']}")
        print(f"  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}  f1={m['f1']:.3f}  yes-rate={m['yes_rate']:.3f}")
        print(f"  confusion: TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}\n")

    if "AV Captioning" in by_task:
        n_cap = len(by_task["AV Captioning"])
        print(f"=== AV Captioning ===")
        print(f"  n={n_cap} (METEOR/CIDEr deferred)\n")


if __name__ == "__main__":
    main()
