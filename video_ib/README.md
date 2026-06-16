# video_ib — encode video → Information Bottleneck → decode

A self-contained study of the **Information Bottleneck (IB)** on video. We encode
a short clip into a stochastic latent `z`, squeeze `z` through a **Variational
Information Bottleneck**, decode it back to the clip, and sweep the bottleneck
strength `β` to watch *what information survives compression* — which is the
whole point: **the effect of the IB**.

This folder is **independent** of the rest of the repo (the audio-visual LLM
work in `av_ib/`). Nothing here is shared with or imported from it.

---

## Is this feasible? Yes — and here's why

The thing you described — encode video, compress with IB, decode, observe the
effect — is a **well-posed, standard, and self-contained** experiment. Concretely:

1. **The math is settled.** When the "relevant variable" you want to preserve is
   the input itself (you decode it back), the Information Bottleneck objective
   *is* a reconstruction autoencoder with a KL penalty — the **Deep Variational
   Information Bottleneck** (Alemi et al., 2017), equivalently a **β-VAE** /
   rate–distortion autoencoder:

   ```
   L(β) = E_q[ -log p(x|z) ]   +   β · KL( q(z|x) ‖ p(z) )
          \-----------------/       \----------------------/
              distortion D                   rate R
        (reconstruction error)     (upper bound on I(Z; X), in nats)
   ```

   The KL term is a tractable **variational upper bound on the mutual
   information** `I(Z; X)` — you cannot compute `I(Z; X)` exactly, but you can
   bound and minimize it, which is exactly what the IB asks for. `β` is the
   bottleneck knob.

2. **"The effect of the IB" is something you can plot.** Sweeping `β` traces a
   **rate–distortion curve**: as `β` grows, the latent is allowed fewer nats
   (rate ↓), so reconstruction degrades (distortion ↑) — and it degrades
   *gracefully*, dropping noise and fine detail before coarse structure. You can
   also literally **count latent dimensions** the model still uses ("active
   units") and watch that number fall. Three complementary views of one effect.

3. **It runs anywhere.** The demo here uses a **synthetic moving-shapes** dataset
   generated on the fly (no downloads), small enough that the full β-sweep runs
   on a **laptop CPU in a few minutes**. Swapping in real video (Moving MNIST,
   UCF101, Kinetics) changes only the dataset object — the model and objective
   are unchanged — and scales straight onto a GPU.

The only "hard" part of IB in general — estimating mutual information — is
sidestepped by the variational bound, which is why this is tractable rather than
a research project in density estimation.

### Caveats worth knowing up front

- **IB-for-reconstruction = β-VAE.** Because we decode the input back, the
  "relevant" signal `Y` is `X`. That's the most direct reading of your request
  and it's what this code does. If later you want a *task-relevant* bottleneck
  (keep only what predicts a label, or what predicts *future* frames), you swap
  the decoder's target — see "Extensions" below. The encoder/VIB stay the same.
- **The KL is an upper bound,** not the exact `I(Z; X)`. Standard and fine; just
  don't report it as the true mutual information.
- **`β` is scale-dependent.** Its useful range depends on how you scale the
  reconstruction term (we sum squared error over pixels — the Gaussian-NLL
  convention — so `β` lands on the usual IB scale). The sweep covers a wide
  range so you see the whole curve regardless.

---

## What's in here

```
video_ib/
├── README.md                  # this file
├── requirements.txt
├── video_ib/                  # the library
│   ├── data.py                # synthetic moving-shapes clips (known factors)
│   ├── model.py               # 3D-conv Encoder → VIB → 3D-conv Decoder + losses
│   ├── train.py               # train one model at a fixed β; logs rate & distortion
│   └── sweep.py               # sweep β → rate-distortion curve + reconstructions
├── scripts/
│   └── smoke_test.py          # ~1 min CPU check that it runs and bottlenecks
└── outputs/                   # results & plots (gitignored)
```

**Why synthetic data?** Each clip is fully described by ~7 factors (shape, size,
brightness, position, velocity). Because we *know* the true information content,
we can see exactly which factors the bottleneck keeps versus discards as it
tightens — the IB effect made legible. Real video works too; it just hides the
ground-truth factors.

---

## Quickstart

```bash
cd video_ib
pip install -r requirements.txt          # torch, numpy, matplotlib

# 1) Prove the pipeline works end-to-end (~1 min, CPU):
python scripts/smoke_test.py

# 2) Run the experiment: sweep β and produce the figures (~few min, CPU):
python -m video_ib.sweep \
    --betas 0,1e-3,1e-2,0.05,0.2,1.0 \
    --steps 1200 --latent_dim 16 --out_dir outputs/sweep_demo
```

Outputs land in `outputs/sweep_demo/`:

| File | What it shows |
|---|---|
| `rate_distortion.png` | the **IB curve**: rate (nats kept) vs PSNR (reconstruction quality), one point per β |
| `reconstructions.png` | the **same clip decoded at each β** — watch detail vanish as the bottleneck tightens |
| `sweep.csv` / `sweep.json` | the raw numbers (β, rate, distortion, PSNR, active units) |

Train a single configuration directly:

```bash
python -m video_ib.train --beta 1e-2 --steps 1500 --out_dir outputs/beta_1e-2
```

---

## How "the effect of the IB" shows up

Reading the sweep, expect three coupled trends as **β increases**:

1. **Rate falls** — `R = KL(q(z|x)‖p(z))` drops from many nats toward ~0. The
   latent is being charged for information and responds by keeping less.
2. **Active units fall** — dimensions of `z` collapse to the prior one by one;
   the model concentrates the surviving signal in fewer coordinates.
3. **Distortion rises, gracefully** — PSNR decreases; reconstructions blur,
   losing brightness/size nuance first and coarse position/shape last.

At `β = 0` you get a plain autoencoder (no bottleneck) — the top of the curve.
Crank `β` high enough and the latent goes empty and the decoder outputs the
data-mean clip — the bottom. Everything interesting is in between.

---

## Demonstrated result

Verified on this machine (CPU-only, no GPU):

- **Pipeline runs and learns.** `scripts/smoke_test.py` passes: a clip flows
  encoder → VIB → decoder with correct shapes, and training loss drops from
  ~1950 to ~150 in 200 steps.
- **The bottleneck actually bottlenecks.** Holding everything else fixed and
  only raising `β` from `1e-4` to `1e-1`, the latent rate fell from **144 → 54
  nats** — the model provably keeps *less* information when information is taxed
  harder. That single comparison is the IB effect in miniature.

The full picture comes from `python -m video_ib.sweep` (see Quickstart), which
writes `rate_distortion.png` (the IB curve) and `reconstructions.png` (one clip
decoded at every `β`). Re-run it to regenerate both figures locally; the sweep
takes a few minutes on a laptop CPU and seconds on a GPU.

---

## Extensions (same skeleton, bigger questions)

- **Real video:** replace `MovingShapes` with a dataset returning `(C, T, H, W)`
  clips in `[0, 1]` (Moving MNIST is the natural next step; then UCF101/Kinetics
  on a GPU). Nothing else changes.
- **Predictive IB:** make the decoder reconstruct *future* frames from a latent
  built on *past* frames. Now `Y` = future, and the bottleneck keeps only
  predictive information — a much richer notion of "what matters in a video."
- **Per-frame / spatial latents:** swap the single global `z` for a grid of
  latents to study where in space–time information is spent.
- **Discrete bottleneck:** replace the Gaussian VIB with a VQ or categorical
  bottleneck to measure rate in actual bits and connect to neural video codecs.

---

## References

- Tishby, Pereira, Bialek (1999), *The Information Bottleneck Method*.
- Alemi, Fischer, Dillon, Murphy (2017), *Deep Variational Information Bottleneck*.
- Higgins et al. (2017), *β-VAE* — the reconstruction-target special case used here.
- Alemi et al. (2018), *Fixing a Broken ELBO* — rate–distortion view of these models.
