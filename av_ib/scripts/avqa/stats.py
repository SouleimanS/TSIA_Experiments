#!/usr/bin/env python3
"""Neat statistics from AVQA eval outputs.

Accuracy models  : reads each <model>/results.jsonl, parses the letter answer
                   (A-D), joins to the annotations to recover the question
                   relation (View / Sound / Both), reports overall + per-relation.
Abstention models: reads <model>_riskcoverage.json, computes coverage and risk
                   per corruption (grouped train / held-out / negative-control),
                   plus a per-relation coverage breakdown.

Run on the cluster where runs/avqa/eval/ and the annotations live:

    python scripts/avqa/stats.py \
        --eval-dir runs/avqa/eval \
        --ann ~/SOULEIMAN_repo/datasets/AVQA/val_qa.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TRAIN_CORRS   = ["identity", "vid_zero", "vid_mean"]
HELDOUT_CORRS = ["vid_noise", "vid_scale"]
NEG_CORRS     = ["aud_zero", "aud_mean"]


def parse_letter(raw) -> str:
    if not raw or str(raw).startswith("<ERROR"):
        return "?"
    m = re.search(r"\b([A-Da-d])\b", str(raw))
    return m.group(1).lower() if m else "?"


# ---------------------------------------------------------------------------
# annotation join: (video_name, question_text) -> question_relation
# ---------------------------------------------------------------------------
def load_relation_map(ann_path: Path) -> dict:
    """AVQA only: (video_name, question_text) -> question_relation.

    AVHBench's qa.json has no such fields; callers pass --ann only for AVQA, so
    an empty map is returned for any annotation file lacking these keys and the
    per-relation breakdown simply collapses to a single overall bucket.
    """
    recs = json.loads(ann_path.read_text())
    m = {}
    for r in recs:
        if "video_name" not in r or "question_text" not in r:
            return {}
        key = (r["video_name"], r["question_text"].strip())
        m[key] = r.get("question_relation", "?")
    return m


def relation_of(rec: dict, relmap: dict) -> str:
    vid = rec.get("video_id")
    q   = rec.get("question", "")
    qtext = q.split("\n")[0].strip() if q else ""
    return relmap.get((vid, qtext), "?")


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------
def accuracy_stats(model_dir: Path, relmap: dict) -> dict | None:
    jsonl = model_dir / "results.jsonl"
    if not jsonl.exists():
        return None
    recs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]

    n_correct = 0
    by_rel = {}
    nlls, kl_vs = [], []
    n_err = 0
    for r in recs:
        gold = str(r.get("gold", "")).strip().lower()
        # Trust the eval's own parsed prediction / correctness (produced by the
        # dataset-aware parser). Recompute from raw text only if both absent.
        if "correct" in r:
            ok = int(r["correct"])
        elif "pred" in r:
            ok = int(str(r["pred"]).strip().lower() == gold)
        else:
            ok = int(parse_letter(r.get("raw_pred", r.get("raw", ""))) == gold)
        if str(r.get("raw_pred", r.get("raw", ""))).startswith("<ERROR"):
            n_err += 1
        n_correct += ok
        rel = relation_of(r, relmap)
        d = by_rel.setdefault(rel, [0, 0])
        d[0] += ok
        d[1] += 1
        if r.get("nll") is not None and r["nll"] == r["nll"]:
            nlls.append(r["nll"])
        if r.get("kl_v") is not None and r["kl_v"] == r["kl_v"]:
            kl_vs.append(r["kl_v"])

    n = len(recs)
    return {
        "n": n,
        "n_correct": n_correct,
        "n_err": n_err,
        "acc_pct": round(100 * n_correct / max(n, 1), 1),
        "mean_nll": round(sum(nlls) / len(nlls), 4) if nlls else None,
        "mean_kl_v": round(sum(kl_vs) / len(kl_vs), 1) if kl_vs else None,
        "per_relation": {
            k: {"acc_pct": round(100 * v[0] / max(v[1], 1), 1), "n": v[1]}
            for k, v in sorted(by_rel.items())
        },
    }


# ---------------------------------------------------------------------------
# risk-coverage
# ---------------------------------------------------------------------------
def riskcov_stats(rc_path: Path, relmap: dict) -> dict:
    data = json.loads(rc_path.read_text())
    per_sample = data.get("per_sample", {})

    out = {"per_corr": {}, "per_corr_relation": {}}
    for corr, recs in per_sample.items():
        n = len(recs)
        n_abst = sum(r.get("abstained", False) for r in recs)
        n_ans  = n - n_abst
        n_cor  = sum(r.get("answered_correct", False) for r in recs)
        out["per_corr"][corr] = {
            "n": n,
            "coverage_pct": round(100 * n_ans / max(n, 1), 1),
            "risk_pct": round(100 * (1 - n_cor / max(n_ans, 1)), 1) if n_ans else None,
            "n_abstained": n_abst,
            "n_answered": n_ans,
            "n_correct": n_cor,
        }
        # per-relation coverage (identity only — the operating point)
        if corr == "identity":
            rel_cov = {}
            for r in recs:
                rel = r.get("meta", {}).get("question_relation", "?")
                d = rel_cov.setdefault(rel, [0, 0])  # [answered, total]
                d[0] += int(not r.get("abstained", False))
                d[1] += 1
            out["per_corr_relation"] = {
                k: {"coverage_pct": round(100 * v[0] / max(v[1], 1), 1), "n": v[1]}
                for k, v in sorted(rel_cov.items())
            }
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="runs/avqa/eval")
    ap.add_argument("--ann", default=None,
                    help="AVQA val_qa.json for the View/Sound/Both relation join "
                         "(optional; omit for AVHBench or any non-AVQA eval)")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    relmap = load_relation_map(Path(args.ann).expanduser()) if args.ann else {}

    print("=" * 64)
    print("ACCURACY  (4-way multiple choice, letter A-D)")
    print("=" * 64)
    print(f"  {'model':<22} {'acc':>7} {'nll':>8} {'kl_v':>9}   per-relation")
    for model_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        st = accuracy_stats(model_dir, relmap)
        if st is None:
            continue
        rels = "  ".join(f"{k}:{v['acc_pct']}%(n={v['n']})"
                         for k, v in st["per_relation"].items())
        err = f"  [!{st['n_err']} gen-errors]" if st.get("n_err") else ""
        print(f"  {model_dir.name:<22} {st['acc_pct']:>6.1f}% "
              f"{st['mean_nll']!s:>8} {st['mean_kl_v']!s:>9}   {rels}{err}")

    print("\n" + "=" * 64)
    print("RISK-COVERAGE  (abstention models)")
    print("=" * 64)
    for rc in sorted(eval_dir.glob("*_riskcoverage.json")):
        st = riskcov_stats(rc, relmap)
        print(f"\n  {rc.stem}")
        for group, corrs, note in [
            ("TRAIN (seen)",  TRAIN_CORRS,   "want: cov high on identity, ~0 on vid_*"),
            ("HELD-OUT",      HELDOUT_CORRS, "generalization to unseen corruptions"),
            ("NEG control",   NEG_CORRS,     "want: cov stays high (audio irrelevant)"),
        ]:
            print(f"    {group:<12} [{note}]")
            print(f"      {'corruption':<12} {'cov':>6} {'risk':>6} {'abst':>9}")
            for c in corrs:
                v = st["per_corr"].get(c)
                if not v:
                    continue
                risk = f"{v['risk_pct']}%" if v["risk_pct"] is not None else "  n/a"
                print(f"      {c:<12} {v['coverage_pct']:>5.1f}% {risk:>6} "
                      f"{v['n_abstained']:>3}/{v['n']:<3}")
        rel = st.get("per_corr_relation", {})
        if rel:
            cov = "  ".join(f"{k}:{v['coverage_pct']}%(n={v['n']})" for k, v in rel.items())
            print(f"    identity coverage by relation: {cov}")

    print("\ndone.")


if __name__ == "__main__":
    main()
