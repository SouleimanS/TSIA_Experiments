"""KL term study for v6 SinkAwareVIB.

Decomposes the KL into its two components and shows how they scale with
encoder output magnitude, so you can choose a sensible beta range before
running Gate 3.

KL per element = 0.5*(mu² + exp(logvar) - logvar - 1)
              = kl_mu_term + kl_var_term

At init (fc_mu=0, logvar_bias=-3):
  mu = x  ->  kl_mu_term  = 0.5 * x²  (large; scales with encoder norm²)
  logvar=-3  ->  kl_var_term = 0.5*(exp(-3)+3-1) ≈ 1.025 per element  (small)

So kl_v at init is dominated by the location term, not the variance.
As beta is raised and fc_mu learns, the mu term shrinks toward 0.

The script also computes the beta sensitivity: the beta at which
  beta * kl = reference_nll
so you can set Gate 3 pilots at sensible grid points.

Usage:
    python -m av_ib.eval.kl_study \\
        --ann-path  $HOME/.../avqa-train.json \\
        --video-root $HOME/.../videos/all \\
        --num-samples 50 \\
        --reference-nll 2.0      # NLL from your baseline pilot log
"""
from __future__ import annotations

import argparse
import math
import statistics as st

import torch


def main(args):
    from av_ib.model.av_model_v6 import AVModelV6
    from av_ib.data.musicavqa import MusicAVQADataset

    print(f"Building AVModelV6 (variant={args.variant}, no LoRA, no checkpoint)...", flush=True)
    model = AVModelV6(use_lora=False, variant=args.variant).eval()
    model.set_sample_noise(False)  # z = mu; isolates KL from reparam noise

    ds = MusicAVQADataset(args.ann_path, args.video_root)
    n = min(args.num_samples, len(ds))
    print(f"Analysing {n} samples (z=mu, no reparam noise)...\n", flush=True)

    # Per-sample accumulators
    kl_v_totals, kl_mu_v, kl_var_v = [], [], []
    kl_sink_v, kl_nonsink_v = [], []
    kl_per_tok_max_v, kl_per_tok_p90_v = [], []
    kl_a_totals = []
    x_norm_means_v = []      # mean ||x||₂ per token (before VIB)
    sink_fracs = []
    n_tokens_v_list = []

    for i in range(n):
        rec = ds[i]
        # Patch bottleneck_v to capture per-token KL components
        _capture = {}

        _orig_fwd = model.bottleneck_v.forward.__func__

        def _patched_fwd(self_, x):
            from av_ib.model.av_model_v6 import _rmsnorm
            normed = _rmsnorm(x)
            phi = normed[..., self_.dsink_dims].abs().max(dim=-1).values
            sink_mask = phi >= self_.tau

            mu_nonsink   = x + self_.fc_mu_nonsink(x)
            logvar_nonsink = self_.fc_logvar_nonsink(x).clamp(-10.0, 10.0)
            mu_sink      = x + self_.fc_mu_sink(x)
            logvar_sink  = (2.0 * self_.log_sigma_sink).expand_as(mu_sink).clamp(-10.0, 10.0)

            mask_3d = sink_mask.unsqueeze(-1)
            mu      = torch.where(mask_3d, mu_sink,      mu_nonsink)
            logvar  = torch.where(mask_3d, logvar_sink,  logvar_nonsink)

            z = mu  # noise off

            # Per-element KL terms
            mu_sq   = mu.pow(2)
            var     = logvar.exp()
            kl_elem_mu  = 0.5 * mu_sq                       # location term
            kl_elem_var = 0.5 * (var - logvar - 1.0)        # variance term
            kl_per_token = (kl_elem_mu + kl_elem_var).mean(dim=-1)  # (B, T)

            _capture["x"]            = x.detach().float()
            _capture["mu"]           = mu.detach().float()
            _capture["logvar"]       = logvar.detach().float()
            _capture["kl_elem_mu"]   = kl_elem_mu.detach().float()
            _capture["kl_elem_var"]  = kl_elem_var.detach().float()
            _capture["kl_per_token"] = kl_per_token.detach().float()   # (B, T)
            _capture["sink_mask"]    = sink_mask.detach()

            # Call original logic for KL combination (to keep last_stats consistent)
            kl_sink    = (kl_per_token *  sink_mask.float()).sum()
            kl_nonsink = (kl_per_token * (~sink_mask).float()).sum()
            kl_combined = kl_nonsink + self_.beta_sink_ratio * kl_sink

            with torch.no_grad():
                phi_f = phi.flatten().float()
                qs = torch.tensor([0.50, 0.60, 0.70, 0.90], device=phi_f.device)
                phi_q = torch.quantile(phi_f, qs)
                self_.last_stats = {
                    "sink_frac":   sink_mask.float().mean().detach().float(),
                    "kl_sink":     kl_sink.detach().float(),
                    "kl_nonsink":  kl_nonsink.detach().float(),
                    "std_nonsink": torch.exp(0.5 * logvar_nonsink).mean().detach().float(),
                    "phi_mean":    phi_f.mean(),
                    "phi_max":     phi_f.max(),
                    "phi_p50":     phi_q[0],
                    "phi_p60":     phi_q[1],
                    "phi_p70":     phi_q[2],
                    "phi_p90":     phi_q[3],
                    "kl_mu_term":  kl_elem_mu.mean().float(),
                    "kl_var_term": kl_elem_var.mean().float(),
                }

            return z, kl_combined, sink_mask

        import types
        model.bottleneck_v.forward = types.MethodType(_patched_fwd, model.bottleneck_v)

        try:
            model.forward_generate(
                [rec["video_path"]], [rec["audio_path"]], [rec["prompt"]],
                max_new_tokens=1,
            )
        except Exception as e:
            print(f"  [{i}] skip: {e}", flush=True)
            model.bottleneck_v.forward = types.MethodType(_orig_fwd, model.bottleneck_v)
            continue

        model.bottleneck_v.forward = types.MethodType(_orig_fwd, model.bottleneck_v)

        if not _capture:
            continue

        x     = _capture["x"]           # (1, T, D)
        kl_pt = _capture["kl_per_token"] # (1, T)
        sm    = _capture["sink_mask"]    # (1, T)
        km    = _capture["kl_elem_mu"]   # (1, T, D)
        kv_   = _capture["kl_elem_var"]  # (1, T, D)

        T = x.shape[1]
        n_tokens_v_list.append(T)

        # Mean over all tokens and dims
        kl_mu_v.append(float(km.mean()))
        kl_var_v.append(float(kv_.mean()))
        kl_v_totals.append(float(kl_pt.sum()))  # total KL (summed over tokens)

        # Sink/non-sink split
        s_v = getattr(model.bottleneck_v, "last_stats", {})
        kl_sink_v.append(float(s_v.get("kl_sink", 0.0)))
        kl_nonsink_v.append(float(s_v.get("kl_nonsink", 0.0)))
        sink_fracs.append(float(s_v.get("sink_frac", float("nan"))))

        # Per-token KL stats
        kl_flat = kl_pt.flatten().float()
        kl_per_tok_max_v.append(float(kl_flat.max()))
        kl_per_tok_p90_v.append(float(torch.quantile(kl_flat, 0.9)))

        # x norm
        x_norm_means_v.append(float(x.norm(dim=-1).mean()))

        # Audio KL
        sa = getattr(model.bottleneck_a, "last_stats", {})
        kla = sa.get("kl_nonsink", None)
        if kla is not None and float(kla) == float(kla):
            kl_a_totals.append(float(kla))

        if (i + 1) % args.every == 0:
            print(f"  [{i+1}/{n}]  kl_v={kl_v_totals[-1]:.1f}  "
                  f"kl_mu={kl_mu_v[-1]:.4f}/elem  kl_var={kl_var_v[-1]:.4f}/elem  "
                  f"sink_frac={sink_fracs[-1]:.3f}", flush=True)

    if not kl_v_totals:
        print("No valid samples — aborting.")
        return

    def _s(lst): return f"{st.mean(lst):.4f}" if lst else "N/A"

    print("\n" + "="*65)
    print("KL STUDY REPORT")
    print("="*65)
    print(f"  samples:               {len(kl_v_totals)}")
    print(f"  avg video tokens (T):  {st.mean(n_tokens_v_list):.0f}")

    print("\n--- KL decomposition (per element, i.e. per (token, dim)) ---")
    print(f"  kl_mu_term    (0.5*mu²):               {_s(kl_mu_v)}")
    print(f"  kl_var_term   (0.5*(σ²-logσ²-1)):      {_s(kl_var_v)}")
    if kl_mu_v and kl_var_v:
        ratio = st.mean(kl_mu_v) / max(st.mean(kl_var_v), 1e-9)
        print(f"  ratio mu/var:                          {ratio:.1f}x  "
              f"({'location-dominated' if ratio > 5 else 'mixed'})")

    print("\n--- KL totals (summed over all tokens, per sample) ---")
    print(f"  kl_v total mean:       {_s(kl_v_totals)}")
    print(f"  kl_v sink portion:     {_s(kl_sink_v)}")
    print(f"  kl_v nonsink portion:  {_s(kl_nonsink_v)}")
    if kl_a_totals:
        print(f"  kl_a total mean:       {_s(kl_a_totals)}")

    print("\n--- Per-token KL (video) ---")
    print(f"  sink_frac mean:        {_s(sink_fracs)}")
    print(f"  kl per-token max:      {_s(kl_per_tok_max_v)}")
    print(f"  kl per-token p90:      {_s(kl_per_tok_p90_v)}")

    print("\n--- Encoder output norms ---")
    print(f"  mean ||x||₂ per token: {_s(x_norm_means_v)}")
    if x_norm_means_v:
        avg_norm = st.mean(x_norm_means_v)
        d = 2048
        # Expected KL per element from location term: 0.5 * (norm²/D)
        expected_kl_mu_elem = 0.5 * (avg_norm ** 2) / d
        print(f"  -> predicted kl_mu/elem = 0.5*norm²/D = {expected_kl_mu_elem:.4f}  "
              f"(matches {_s(kl_mu_v)} above?)")

    print("\n--- Beta sensitivity ---")
    ref_nll = args.reference_nll
    avg_kl_v = st.mean(kl_v_totals) if kl_v_totals else float("nan")
    avg_kl_a = st.mean(kl_a_totals) if kl_a_totals else float("nan")
    print(f"  reference NLL:         {ref_nll:.3f}")
    print(f"  beta at which beta*kl_v = NLL:")
    if avg_kl_v > 0:
        beta_eq = ref_nll / avg_kl_v
        print(f"    beta_eq = {ref_nll:.3f} / {avg_kl_v:.1f} = {beta_eq:.2e}")
        print(f"  Suggested Gate 3 grid (decade steps around beta_eq):")
        for mult in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0):
            b = beta_eq * mult
            kl_contrib = b * avg_kl_v
            print(f"    beta = {b:.2e}  ->  beta*kl_v = {kl_contrib:.3f}  "
                  f"({100*kl_contrib/ref_nll:.0f}% of NLL)")

    print("\n--- Interpretation ---")
    if kl_mu_v and kl_var_v and st.mean(kl_mu_v) > 5 * st.mean(kl_var_v):
        print("  The KL is location-dominated (kl_mu >> kl_var).")
        print("  During training, fc_mu will learn to push mu toward 0 for tokens")
        print("  that don't help predict the answer. The variance term is small")
        print("  at init and will grow slightly as logvar learns to be less negative.")
        print("  -> Effective compression = fc_mu shrinks high-norm uninformative tokens.")
    else:
        print("  KL is mixed location+variance. Both paths active.")

    print("\n  To study KL evolution over training: watch kl_mu_term_{v,a} and")
    print("  kl_var_term_{v,a} in the training log (now emitted by last_stats).")
    print("="*65, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann-path", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--variant", default="b_topk_nofusion")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--reference-nll", type=float, default=2.0,
                   help="NLL from baseline pilot (used for beta sensitivity)")
    args = p.parse_args()
    main(args)
