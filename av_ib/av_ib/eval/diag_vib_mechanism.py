#!/usr/bin/env python3
"""Diagnostic 3 — VIB mechanism & loss mathematics.

This is the deep diagnostic. It has TWO modes.

────────────────────────────────────────────────────────────────────────────
MODE A — log analysis (cheap, no GPU): read training log.jsonl files.
────────────────────────────────────────────────────────────────────────────
Reconstructs the loss decomposition over training and reports the things that
actually explain why a run learned or collapsed:

  loss = nll  +  beta_v·KL_v  +  beta_a·KL_a  +  beta_j·KL_j
              +  aux_weight·(nll_aux_v + nll_aux_a)

  - Effective contribution of each term to the total loss over time.
  - KL magnitudes: with beta=0 the KL is UNREGULARIZED — we show how large it
    grows and whether the encoder is exploiting that (rate explosion).
  - NLL trajectory: is the LLM actually fitting the answer (nll→0) or stuck?
  - aux-head trajectory: are the per-modality heads learning (aux nll drop)?
  - grad-norm spikes: instability markers.
  - "posterior collapse" check from logs: KL→0 while nll flat = collapse.

────────────────────────────────────────────────────────────────────────────
MODE B — live probe (needs GPU + checkpoint): load a model and run N records.
────────────────────────────────────────────────────────────────────────────
Measures, on real data, what the math is doing INSIDE the VIB:

  - Rate-distortion point: mean KL-per-token (the "rate" R) for video & audio.
  - Sink-mask rate: fraction of tokens classified as sinks (video φ-based,
    audio norm-top-k). If this is ~0% or ~100%, the sink mechanism is inert.
  - mu/sigma statistics: is sigma collapsing to 0 (deterministic, no IB) or
    blowing up (noise destroying signal)?
  - Per-token KL histogram: a few tokens hoarding all the rate = sink hoarding.
  - z-norm preservation: are sink tokens actually preserved (||z||≈||x||)?

Usage (Mode A — always runnable):
  python diag_vib_mechanism.py --mode logs \
      --log b@1k:runs/v6_tier1_b/log.jsonl \
      --log bbis@1k:results/ckpts/musicavqa_bis_1k/log.jsonl \
      --betas 0,0,0 --aux-weight 0.1 \
      --out-dir results/diagnostics

Usage (Mode B — submit as a GPU job):
  python diag_vib_mechanism.py --mode probe \
      --ckpt-path results/ckpts/musicavqa_bis_1k/final.pt \
      --variant b_bis --audio-vib norm_topk --fusion mutual \
      --ann-path /path/avqa-test.json \
      --video-root /path/videos --num-records 30 \
      --out-dir results/diagnostics
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# MODE A — log analysis
# ══════════════════════════════════════════════════════════════════════════════

def load_log(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("{") is False:
                # skip non-JSON lines (the final summary block may be multi-line)
                pass
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    # keep only per-step records (have a "step" key)
    return [r for r in rows if "step" in r and "loss" in r]


def _series(rows: List[Dict], key: str) -> List[float]:
    return [float(r[key]) for r in rows if key in r and r[key] is not None]


def _window_mean(xs: List[float], frac_lo: float, frac_hi: float) -> float:
    if not xs:
        return float("nan")
    lo = int(len(xs) * frac_lo)
    hi = int(len(xs) * frac_hi)
    seg = xs[lo:hi] or xs
    return sum(seg) / len(seg)


def analyze_log(name: str, rows: List[Dict], betas: Tuple[float, float, float],
                aux_w: float) -> Dict:
    bv, ba, bj = betas
    print("=" * 78)
    print(f"LOSS DECOMPOSITION — {name}  (betas v/a/j = {bv}/{ba}/{bj}, "
          f"aux_w = {aux_w})")
    print("=" * 78)
    if not rows:
        print("  (no per-step records found)")
        return {}

    n = len(rows)
    loss   = _series(rows, "loss")
    nll    = _series(rows, "nll")
    klv    = _series(rows, "kl_v")
    kla    = _series(rows, "kl_a")
    klj    = _series(rows, "kl_j")
    auxv   = _series(rows, "nll_aux_v")
    auxa   = _series(rows, "nll_aux_a")
    gn     = _series(rows, "grad_norm")

    def show(label, xs, beta=1.0):
        if not xs:
            print(f"  {label:<14}: (absent)")
            return None
        first = _window_mean(xs, 0.0, 0.1)
        last  = _window_mean(xs, 0.9, 1.0)
        peak  = max(xs)
        contrib_last = beta * last
        print(f"  {label:<14}: start={first:>12.4f}  end={last:>12.4f}  "
              f"peak={peak:>12.4f}  (β·end = {contrib_last:>10.4f})")
        return {"start": first, "end": last, "peak": peak,
                "beta_contribution_end": contrib_last}

    out = {"n_steps": n}
    print(f"  steps logged   : {n}")
    out["loss"]  = show("loss", loss)
    out["nll"]   = show("nll (main)", nll)
    out["kl_v"]  = show("KL_v", klv, bv)
    out["kl_a"]  = show("KL_a", kla, ba)
    out["kl_j"]  = show("KL_j", klj, bj)
    out["aux_v"] = show("nll_aux_v", auxv, aux_w)
    out["aux_a"] = show("nll_aux_a", auxa, aux_w)
    out["grad_norm"] = show("grad_norm", gn)

    # ── Interpretation flags ─────────────────────────────────────────────
    print("\n  ── interpretation ──")
    flags = []

    # 1. Where does the loss actually come from at the end?
    end_nll  = _window_mean(nll, 0.9, 1.0) if nll else 0.0
    end_klv  = bv * _window_mean(klv, 0.9, 1.0) if klv else 0.0
    end_kla  = ba * _window_mean(kla, 0.9, 1.0) if kla else 0.0
    end_klj  = bj * _window_mean(klj, 0.9, 1.0) if klj else 0.0
    end_aux  = aux_w * ((_window_mean(auxv, 0.9, 1.0) if auxv else 0.0)
                        + (_window_mean(auxa, 0.9, 1.0) if auxa else 0.0))
    total = end_nll + end_klv + end_kla + end_klj + end_aux
    if total > 0:
        print(f"  end-of-training loss composition:")
        print(f"    nll        : {end_nll/total:>6.1%}")
        print(f"    β_v·KL_v   : {end_klv/total:>6.1%}")
        print(f"    β_a·KL_a   : {end_kla/total:>6.1%}")
        print(f"    β_j·KL_j   : {end_klj/total:>6.1%}")
        print(f"    aux        : {end_aux/total:>6.1%}")
        out["end_loss_composition"] = {
            "nll": end_nll / total, "beta_v_kl_v": end_klv / total,
            "beta_a_kl_a": end_kla / total, "beta_j_kl_j": end_klj / total,
            "aux": end_aux / total,
        }

    # 2. Unregularized KL explosion (beta=0 but KL huge)?
    if klv and bv == 0:
        peak_klv = max(klv)
        if peak_klv > 1000:
            flags.append(f"KL_v unregularized & large (peak={peak_klv:.0f}): "
                         f"video encoder is using unbounded rate — VIB is decorative at β_v=0")

    # 3. aux head learning?
    if auxv:
        if _window_mean(auxv, 0.9, 1.0) < 0.5 * _window_mean(auxv, 0.0, 0.1):
            flags.append("aux_v dropped >2x: video aux head IS learning the answer "
                         "from pooled z_v (video carries task signal)")
        else:
            flags.append("aux_v ~flat: video aux head NOT learning — z_v may lack "
                         "linearly-decodable answer info")
    if auxa:
        if _window_mean(auxa, 0.9, 1.0) < 0.5 * _window_mean(auxa, 0.0, 0.1):
            flags.append("aux_a dropped >2x: audio aux head IS learning from z_a")
        else:
            flags.append("aux_a ~flat: audio aux head NOT learning — audio path "
                         "carries little linearly-decodable answer signal")

    # 4. NLL convergence
    if nll:
        if _window_mean(nll, 0.9, 1.0) < 0.1:
            flags.append("nll→~0: LLM fits the answer tokens easily (teacher-forced); "
                         "train loss is NOT a good proxy for eval accuracy here")

    # 5. grad spikes
    if gn:
        spikes = sum(1 for g in gn if g > 10)
        if spikes:
            flags.append(f"{spikes} grad-norm spikes >10: training instability "
                         f"(clip at 1.0 mitigates but signals rough loss surface)")

    for fl in flags:
        print(f"  • {fl}")
    out["flags"] = flags
    print()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MODE B — live probe
# ══════════════════════════════════════════════════════════════════════════════

def run_probe(args) -> Dict:
    import torch
    import random

    print("=" * 78)
    print(f"LIVE VIB PROBE — {args.ckpt_path}  variant={args.variant}")
    print("=" * 78)

    from av_ib.model.av_model_v6 import AVModelV6
    model = AVModelV6(
        use_lora=True, variant=args.variant,
        video_vib=args.video_vib, audio_vib=getattr(args, "audio_vib", "standard"),
        fusion_type=args.fusion,
    )
    model.eval()
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("trainable_state", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded ckpt ({len(state)} tensors; "
          f"{len(missing)} missing, {len(unexpected)} unexpected)")

    # Hook the bottleneck modules to capture mu/logvar/sink_mask.
    captures = {"v": [], "a": []}

    def make_hook(slot):
        def hook(module, inp, out):
            # out is (z, kl, sink_mask) for SinkAware/NormTopK, (z, kl) for VIB
            x = inp[0].detach()
            z = out[0].detach()
            kl = out[1].detach()
            mask = out[2].detach() if len(out) > 2 else None
            captures[slot].append({
                "x_norm": x.norm(dim=-1).flatten().float().cpu(),
                "z_norm": z.norm(dim=-1).flatten().float().cpu(),
                "kl": float(kl),
                "n_tokens": x.shape[1],
                "sink_rate": (float(mask.float().mean()) if mask is not None else None),
                "preserved": (float((z[mask] - x[mask]).norm() / (x[mask].norm() + 1e-9))
                              if mask is not None and mask.any() else None),
            })
        return hook

    h1 = model.bottleneck_v.register_forward_hook(make_hook("v"))
    h2 = model.bottleneck_a.register_forward_hook(make_hook("a"))

    # Sample records
    import re
    _PH = re.compile(r"<[A-Za-z][A-Za-z0-9_]*>")
    def render(qc, templ):
        try: vals = json.loads(templ)
        except Exception: return qc
        out = qc
        for v in vals: out = _PH.sub(str(v), out, count=1)
        return out

    with open(args.ann_path) as f:
        records = json.load(f)
    vroot = Path(args.video_root)
    records = [r for r in records
               if r.get("question_deleted", 0) == 0
               and (vroot / f"{r['video_id']}.mp4").exists()]
    rng = random.Random(args.seed)
    picked = rng.sample(records, min(args.num_records, len(records)))
    print(f"  probing {len(picked)} records\n")

    for i, rec in enumerate(picked):
        vp = str(vroot / f"{rec['video_id']}.mp4")
        prompt = render(rec["question_content"], rec["templ_values"])
        try:
            with torch.inference_mode():
                _ = model.forward_generate([vp], [vp], [prompt])
        except Exception as e:
            print(f"  [skip] {rec['video_id']}: {type(e).__name__}: {e}")

    h1.remove(); h2.remove()

    # Aggregate
    out = {}
    for slot, name in [("v", "VIDEO"), ("a", "AUDIO")]:
        caps = captures[slot]
        if not caps:
            print(f"  {name}: no captures")
            continue
        import torch as _t
        kls = [c["kl"] for c in caps]
        ntok = [c["n_tokens"] for c in caps]
        kl_per_tok = [c["kl"] / max(c["n_tokens"], 1) for c in caps]
        sink_rates = [c["sink_rate"] for c in caps if c["sink_rate"] is not None]
        preserved = [c["preserved"] for c in caps if c["preserved"] is not None]
        all_x = _t.cat([c["x_norm"] for c in caps])
        all_z = _t.cat([c["z_norm"] for c in caps])

        print(f"  ── {name} VIB ──")
        print(f"    mean KL (rate R)        : {sum(kls)/len(kls):.2f} nats")
        print(f"    mean KL per token       : {sum(kl_per_tok)/len(kl_per_tok):.3f}")
        print(f"    mean x-norm (input)     : {all_x.mean():.3f}")
        print(f"    mean z-norm (output)    : {all_z.mean():.3f}")
        print(f"    z/x norm ratio          : {(all_z.mean()/all_x.mean()):.3f}")
        if sink_rates:
            print(f"    sink-mask rate          : {sum(sink_rates)/len(sink_rates):.1%}")
        if preserved:
            print(f"    sink preservation err   : {sum(preserved)/len(preserved):.3f}"
                  f"  (||z-x||/||x|| on sinks; ~0 = preserved)")
        # Collapse diagnostics
        flags = []
        mean_klpt = sum(kl_per_tok) / len(kl_per_tok)
        if mean_klpt < 0.01:
            flags.append("KL/token ≈ 0: POSTERIOR COLLAPSE — VIB passes a point estimate, "
                         "no information bottleneck active")
        if sink_rates:
            sr = sum(sink_rates) / len(sink_rates)
            if sr < 0.02:
                flags.append("sink rate ≈ 0%: sink mechanism INERT (almost no tokens flagged)")
            elif sr > 0.95:
                flags.append("sink rate ≈ 100%: sink mechanism INERT (all tokens flagged)")
        for fl in flags:
            print(f"    • {fl}")
        print()
        out[slot] = {
            "mean_kl": sum(kls) / len(kls),
            "mean_kl_per_token": mean_klpt,
            "mean_x_norm": float(all_x.mean()),
            "mean_z_norm": float(all_z.mean()),
            "z_x_ratio": float(all_z.mean() / all_x.mean()),
            "sink_rate": (sum(sink_rates) / len(sink_rates)) if sink_rates else None,
            "sink_preservation_err": (sum(preserved) / len(preserved)) if preserved else None,
            "flags": flags,
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("logs", "probe"), required=True)
    # logs mode
    p.add_argument("--log", action="append", default=[], metavar="LABEL:PATH")
    p.add_argument("--betas", default="0,0,0", help="beta_v,beta_a,beta_j")
    p.add_argument("--aux-weight", type=float, default=0.1)
    # probe mode
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--variant", default="b")
    p.add_argument("--video-vib", default="sink")
    p.add_argument("--audio-vib", default="standard")
    p.add_argument("--fusion", default="mutual")
    p.add_argument("--ann-path", default=None)
    p.add_argument("--video-root", default=None)
    p.add_argument("--num-records", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/diagnostics")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "logs":
        betas = tuple(float(x) for x in args.betas.split(","))
        summary = {}
        for spec in args.log:
            label, path = spec.split(":", 1)
            rows = load_log(Path(path))
            summary[label] = analyze_log(label, rows, betas, args.aux_weight)
        with open(out_dir / "vib_logs.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] {out_dir / 'vib_logs.json'}")
    else:
        assert args.ckpt_path and args.ann_path and args.video_root, \
            "probe mode needs --ckpt-path --ann-path --video-root"
        summary = run_probe(args)
        with open(out_dir / "vib_probe.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] {out_dir / 'vib_probe.json'}")


if __name__ == "__main__":
    main()
