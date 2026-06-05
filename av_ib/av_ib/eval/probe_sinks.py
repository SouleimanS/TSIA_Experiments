"""Probe: do dim-based sinks ({985,1992}, tau=18) actually fire on the VISION
encoder output?

This answers the thesis-critical question (av_model_v6 finding #5) in minutes
without committing to a training run. The sink dims were validated in the LLM
residual stream; the SinkAwareVIB classifies on encoder output. If sinks don't
exist there, the whole "protect the sinks" mechanism is silently inert.

Runs the untrained model's provider over N samples in eval mode and reports the
distribution of sink_frac_v (fraction of video tokens classified as sinks).

    sink_frac_v ~ 0   -> SinkAwareVIB is inert in encoder space. Relocate the
                         IB into the LLM residual stream, or drop dim-based
                         sinks for norm-top-k everywhere.
    sink_frac_v > 0   -> sinks are real here; proceed to the beta sweep.

Usage:
    python -m av_ib.eval.probe_sinks \\
        --ann-path  $HOME/.../avqa-train.json \\
        --video-root $HOME/.../videos/all \\
        --num-samples 100
"""
from __future__ import annotations

import argparse
import statistics as st

import torch


def main(args):
    from av_ib.model.av_model_v6 import AVModelV6
    from av_ib.data.musicavqa import MusicAVQADataset

    print(f"Building AVModelV6 (variant={args.variant}, untrained, no LoRA)...", flush=True)
    model = AVModelV6(use_lora=False, variant=args.variant).eval()

    ds = MusicAVQADataset(args.ann_path, args.video_root)
    n = min(args.num_samples, len(ds))
    print(f"Probing {n} / {len(ds)} samples...", flush=True)

    fracs_v, fracs_a, n_zero_v = [], [], 0
    for i in range(n):
        rec = ds[i]
        try:
            model.forward_generate(
                [rec["video_path"]], [rec["audio_path"]], [rec["prompt"]],
                max_new_tokens=1,
            )
        except Exception as e:
            print(f"  [{i}] skip: {e}", flush=True)
            continue

        sv = getattr(model.bottleneck_v, "last_stats", {}) or {}
        sa = getattr(model.bottleneck_a, "last_stats", {}) or {}

        fv = float(sv.get("sink_frac", float("nan")))
        if fv == fv:  # not NaN
            fracs_v.append(fv)
            if fv == 0.0:
                n_zero_v += 1
        fa = sa.get("sink_frac", None)
        if fa is not None and float(fa) == float(fa):
            fracs_a.append(float(fa))

        if (i + 1) % args.every == 0 and fracs_v:
            print(f"  [{i+1}/{n}] sink_frac_v mean={st.mean(fracs_v):.4f}  "
                  f"zero={n_zero_v}/{len(fracs_v)}", flush=True)

    if not fracs_v:
        print("No valid samples — aborting.")
        return

    print("\n=== Sink probe results ===")
    print(f"  samples:           {len(fracs_v)}")
    print(f"  sink_frac_v mean:  {st.mean(fracs_v):.4f}")
    print(f"  sink_frac_v range: [{min(fracs_v):.4f}, {max(fracs_v):.4f}]")
    print(f"  samples == 0:      {n_zero_v}/{len(fracs_v)} "
          f"({100*n_zero_v/len(fracs_v):.1f}%)")
    if fracs_a:
        print(f"  sink_frac_a mean:  {st.mean(fracs_a):.4f}  "
              f"(top-k audio control, expect ~{0.40:.2f})")

    verdict = ("INERT — dim-based sinks do not fire in vision-encoder space. "
               "The SinkAwareVIB is just a residual VIB here."
               if st.mean(fracs_v) < 1e-4 else
               "ACTIVE — sinks present in encoder space; proceed to the beta sweep.")
    print(f"\n  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--variant", default="b_topk_fusion",
                   help="Any b variant works (bottleneck_v is SinkAwareVIB in all)")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--every", type=int, default=20)
    args = p.parse_args()
    main(args)
