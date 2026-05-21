"""Plot accuracy-vs-step curves for one or more MUSIC-AVQA runs.

Usage:
    python scripts/plot_curves.py v1_3ep                 # just v1
    python scripts/plot_curves.py v1_3ep v2_3ep_nb1      # compare two
    python scripts/plot_curves.py v1_3ep --overall --modality  # both panels
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, saves PNG without a display
import matplotlib.pyplot as plt


def load_evals(run_name: str):
    p = Path("runs") / f"musicavqa_{run_name}" / "eval_log.jsonl"
    if not p.exists():
        print(f"WARN: {p} not found")
        return None
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Run name suffixes, e.g. v1_3ep")
    ap.add_argument("--out", default="curves.png", help="Output PNG path")
    ap.add_argument("--modality", action="store_true",
                    help="Add per-modality panels (Audio, Audio-Visual, Visual)")
    args = ap.parse_args()

    runs_data = {r: load_evals(r) for r in args.runs}
    runs_data = {k: v for k, v in runs_data.items() if v}
    if not runs_data:
        print("No data found"); return

    n_panels = 4 if args.modality else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    # Panel 0: overall
    ax = axes[0]
    for name, evals in runs_data.items():
        steps = [e["step"] for e in evals]
        acc = [e["accuracy"] for e in evals]
        ax.plot(steps, acc, marker="o", label=name)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Overall accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    if args.modality:
        for i, mod in enumerate(["Audio", "Audio-Visual", "Visual"], start=1):
            ax = axes[i]
            for name, evals in runs_data.items():
                steps, accs = [], []
                for e in evals:
                    pm = e.get("per_modality", {})
                    if mod in pm:
                        steps.append(e["step"])
                        accs.append(pm[mod]["acc"])
                if steps:
                    ax.plot(steps, accs, marker="o", label=name)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{mod} (small n for Audio/Visual)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
