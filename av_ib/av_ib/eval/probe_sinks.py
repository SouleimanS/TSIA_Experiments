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
    # phi distribution accumulators (per-sample summaries of the video sink score)
    phi_means, phi_maxes, phi_p50, phi_p60, phi_p70, phi_p90 = [], [], [], [], [], []
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

        # Collect phi summaries when present (SinkAwareVIB only)
        def _g(k):
            v = sv.get(k, None)
            return float(v) if v is not None else None
        for acc, key in ((phi_means, "phi_mean"), (phi_maxes, "phi_max"),
                         (phi_p50, "phi_p50"), (phi_p60, "phi_p60"),
                         (phi_p70, "phi_p70"), (phi_p90, "phi_p90")):
            val = _g(key)
            if val is not None and val == val:
                acc.append(val)

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

    # --- phi distribution (the sink score before thresholding at tau) ---
    cur_tau = float(model.bottleneck_v.tau)
    if phi_means:
        print(f"\n=== phi distribution (video sink score; current tau={cur_tau:.1f}) ===")
        print(f"  phi mean:  {st.mean(phi_means):.3f}")
        print(f"  phi max:   {st.mean(phi_maxes):.3f}  (avg per-sample max)")
        print(f"  phi p50:   {st.mean(phi_p50):.3f}")
        print(f"  phi p60:   {st.mean(phi_p60):.3f}")
        print(f"  phi p70:   {st.mean(phi_p70):.3f}")
        print(f"  phi p90:   {st.mean(phi_p90):.3f}")
        # A tau between p60 and p70 yields a 30-40% sink fraction, matching the
        # audio top-k control. Recommend p70 as a starting point if inert.
        rec_tau = st.mean(phi_p70)
        print(f"  -> if recalibrating, try tau ~ {rec_tau:.1f} (p70 -> ~30% sinks)")

    mean_v = st.mean(fracs_v)
    if mean_v < 1e-4:
        verdict = ("INERT — dim-based sinks do not fire at tau={:.1f} in "
                   "vision-encoder space. SinkAwareVIB is just a residual VIB. "
                   "Options: (1) recalibrate tau to the p70 above; (2) switch "
                   "video to NormTopKSinkVIB; (3) relocate the IB into the LLM "
                   "residual stream.".format(cur_tau))
    elif mean_v > 0.9:
        verdict = ("SATURATED — nearly every token classifies as a sink at "
                   "tau={:.1f}. Raise tau toward the p90 above so sinks are a "
                   "minority.".format(cur_tau))
    else:
        verdict = ("ACTIVE — sinks present in encoder space (sink_frac_v="
                   "{:.3f}); proceed to the beta sweep.".format(mean_v))
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
