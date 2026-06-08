# ROADMAP — AV-LLM Hallucination Mitigation via Information Bottleneck

**Date:** 2026-06-08
**Branch:** `claude/intelligent-gauss-frySl`
**Status:** Strategic reset after end-to-end code + literature review. This document
supersedes the ad-hoc "Gate 1/2/3" plan. It is written to be executed step by step.

---

## 0. TL;DR (read this first)

1. **The mechanism has precedent and it works — but not the way v6 does it.**
   AdaVIB (AAAI 2025) puts a VIB on visual tokens before the LLM and cuts
   hallucination on POPE. It works because it is **adaptive on query/answer
   relevance** (noise the irrelevant features, keep the relevant ones) and is
   **evaluated directly on hallucination**, not on clean-QA NLL.

2. **The "sink-aware" twist is probably backwards.** The literature (incl.
   *Visual Attention Sink in LMMs*, ICLR 2025) says sink tokens are
   low-information attention "valves": removing them barely hurts; removing
   random tokens hurts a lot. `SinkAwareVIB` protects sinks (100× less KL) and
   compresses non-sinks — i.e. it preserves the junk and squeezes the signal.
   This mechanistically explains why VIB has hurt in every prior generation
   (README: v1 0.680 → v3 0.636; v5 β=0 was 25 pts below baseline).

3. **The current gate loop cannot detect success or failure.** It tunes β on
   MUSIC-AVQA *training* NLL, which is (a) floored by memorization (2k samples ×
   2k steps → NLL≈4e-4), (b) gameable by language priors, and (c) confounded by
   620M-param aux heads that leak the answer into `z`. The real metric —
   AVHBench hallucination F1 — is never in the loop.

4. **The honest, defensible thesis:** *"When does an information bottleneck on
   audio-visual encoder outputs reduce cross-modal hallucination in Qwen3-Omni,
   and when it doesn't, why?"* This wins either way. The explainability
   experiments are the thesis instrument, not a side quest.

5. **What to actually do (order):** fix the confounds (Phase 0) → build the
   explainability/measurement instrument (Phase 1) → fix the eval harness
   (Phase 2) → run ONE clean, relevance-adaptive experiment with the right
   metric and ≥3 seeds (Phase 3) → scale or write the negative result (Phase 4).

---

## 1. Where we actually are

### 1.1 The architecture (verified from code)

Qwen3-Omni-30B-A3B-Instruct, frozen, LoRA (r=16) on `q/k/v/o + linear_fc1/fc2`
inside the Thinker text model. The C-MIB modules intercept encoder outputs
mid-forward via hooks (`backbone/splicing.py`):

```
video frames ─► visual encoder ─► pooler_output  (N_v×2048) ─┐
audio        ─► audio encoder  ─► last_hidden     (N_a×2048) ─┤
                                                              ▼
                              CMIBSplicer captures both, calls provider():
                                bottleneck_v(video)  → z_v, kl_v   [SinkAwareVIB]
                                bottleneck_a(audio)  → z_a, kl_a   [VIB or NormTopKSinkVIB]
                                (optional) fusion(z_v, z_a)        [MutualCrossAttention]
                                (variants a/c) bottleneck_joint
                              z_joint = cat([z_v, z_a])
                                                              ▼
                       masked_scatter z_joint into inputs_embeds at the
                       audio/video placeholder positions, then the frozen+LoRA
                       LLM runs as normal.
```

Loss (`train/loop.py:38`):
`loss = nll + β_v·kl_v + β_a·kl_a + β_j·kl_j + aux_w·(nll_aux_v + nll_aux_a)`,
fixed β from step 0 (no warmup), AdamW lr=1e-4, **batch size 1**.

### 1.2 What v6 got right

The zero-init residual VIB (`bottleneck.py:76`), zero-init fusion
(`fusion.py:50`), and `sample_noise` gate mean the model is a **true identity at
step 0** — β=0 + noise-off ≈ vanilla Qwen. That removed the mechanical reasons
v5 started 25 pts below baseline. **This is the real accomplishment of v6: a
clean platform.** Everything below is about using it correctly.

### 1.3 The literature verdict (two independent searches)

**IB-for-hallucination — precedent exists, frame novelty carefully:**
- **AdaVIB** (AAAI 2025, [2502.20750](https://arxiv.org/abs/2502.20750)): VIB on
  projected visual tokens before the LLM, entropy-**adaptive** noise; ~7.5–8%
  POPE gains on LLaVA. *Closest precedent — your `_adavib` variant points at it.*
- **VIBRA** (OpenReview `dT2gGWoO6m`): redundancy-aware multimodal IB on visual
  features, query-aware gate.
- **Vittle** (NeurIPS 2025, [2505.13946](https://arxiv.org/abs/2505.13946)): IB
  *inside* the LLM for MLLM robustness.
- **Foundations:** Tishby IB ([1503.02406](https://arxiv.org/pdf/1503.02406)),
  Deep VIB (Alemi et al. ICLR 2017, [1612.00410](https://arxiv.org/abs/1612.00410)),
  Achille-Soatto invariance ([JMLR 2018](https://jmlr.org/papers/volume19/17-646/17-646.pdf)).
- **AVHBench** (ICLR 2025, [2410.18325](https://arxiv.org/abs/2410.18325)):
  4 tasks (Audio-driven Video Hall., Video-driven Audio Hall., AV Matching,
  AV Captioning). Binary → Acc/Prec/Rec/**F1**; captioning → METEOR/CIDEr/GAVIE.
- **Verdict:** "VIB on encoder outputs before the LLM" is **not novel** (AdaVIB).
  **Novel & defensible:** the **audio-visual / cross-modal** setting,
  bottlenecking **both** encoders, **AVHBench** eval, and a *correct* relevance
  mechanism. Cite AdaVIB/VIBRA/Vittle as the visual-only precedents you extend.

**Attention sinks — the preservation premise is contradicted:**
- StreamingLLM (ICLR 2024, [2309.17453](https://arxiv.org/abs/2309.17453)): sinks
  absorb excess softmax mass; it's the **position/KV** that matters, not content.
- Massive Activations (Sun et al., COLM 2024,
  [2402.17762](https://arxiv.org/abs/2402.17762)): few input-agnostic massive
  dims *cause* sinks; **indices are model-specific** (LLaMA2-7B `{1415,2533}`),
  detected as **>1000× median**, not a bare threshold.
- **Visual Attention Sink in LMMs** (ICLR 2025,
  [2503.03321](https://arxiv.org/abs/2503.03321)): *"removing irrelevant visual
  sink tokens does not impact model performance"*; masking sinks barely hurts,
  masking random tokens hurts sharply. Qwen2-VL-7B sinks at `{458,2570}`.
- ViT Registers (ICLR 2024, [2309.16588](https://arxiv.org/abs/2309.16588)):
  high-norm artifact tokens are discardable "scratchpads."
- "Attend to first token" (ACL 2025, [2504.02732](https://arxiv.org/abs/2504.02732)),
  "When Attention Sink Emerges" (ICLR 2025 Spotlight,
  [2410.10781](https://arxiv.org/abs/2410.10781)): sinks are **non-informative**
  no-ops that prevent over-mixing.
- **Implication for us:** sink-vs-non-sink is **not** the compress-vs-preserve
  axis. Protecting sinks while compressing non-sinks throws away signal. The
  `{985,1992}` dims match no published model and **must be re-validated by causal
  ablation on Qwen3-Omni**, not assumed.

---

## 2. The math, corrected

### 2.1 The objective (what we should be optimizing)

Deep Variational IB (Alemi 2017): learn a stochastic encoding `Z` of input `X`
that is maximally predictive of target `Y`, minimally informative about `X`:

```
  max  I(Z;Y) − β·I(X;Z)
```
- `I(Z;Y)` is lower-bounded by `E[log q(y|z)]` → the LLM's answer **NLL**.
- `I(X;Z)` is upper-bounded by `E_x KL(p(z|x) ‖ r(z))`, prior `r(z)=N(0,I)`.

So `loss = NLL + β·KL(p(z|x)‖N(0,I))`. The code's structure is correct. The
problems are in the *details* and the *axis of selectivity*.

### 2.2 Problem A — the prior pulls z→0, and with floored NLL it just erases tokens

KL per element `= 0.5·(μ² + σ² − log σ² − 1)`. The gradient of the location term
`0.5·μ²` w.r.t. `μ` is `μ` — it pulls `μ` toward **0** (the prior mean). With the
residual `μ = x + f(x)`, that trains `f(x) → −x`, i.e. **the network learns to
erase the encoder token.** That *is* compression — fine in theory.

But selectivity (compress nuisance, keep signal) only exists if NLL **pushes
back** on useful tokens. In the current pilots NLL is floored at ~4e-4
(memorization + language prior), so there is almost no counter-gradient: β just
shrinks **everything uniformly**. That is the mechanical reason "VIB hurts" on
clean, memorized QA. **Fix:** measure on a metric that is *not* floored
(held-out + hallucination), and make selectivity explicit (§2.5).

### 2.3 Problem B — the sink variance-KL is a self-inflicted init artifact

`kl_study` found `kl_var=1.94/elem ≫ kl_mu=0.073/elem`. The variance term is
dominated by sink tokens: `log_sigma_sink = −10` → `logvar = −20` (clamped −10) →
`−log σ²/2 ≈ 5` per sink element. "Preserve the sink" means low noise (`σ→0`),
which **maximizes** `−log σ²` → maximizes KL. So `beta_sink_ratio=0.01` is a patch
over a penalty we created, and the loss-minimizing direction *raises* sink σ
(adds noise to sinks) — the opposite of preservation.

**Fix (whichever sink design survives §2.5):** tokens you intend to *preserve*
should **bypass the KL entirely** (pure passthrough, `z=x`, no variance term),
not be handed a near-zero σ that the KL then fights.

### 2.4 Problem C — KL scales with token count (β is not comparable)

`SinkAwareVIB`/`NormTopKSinkVIB` reduce KL with `.sum()` over tokens, and video
is ~14k tokens (uncapped Qwen frame sampling). So total KL ∝ N_tokens, and β
pressure varies ~20× across samples of different video length. **Fix:** reduce KL
as a **per-token mean** so β is a per-token *rate* comparable across samples and
to the literature. Re-derive the β grid afterwards (the current
`beta_eq=6.93e-5` is computed against the summed KL=28840 and is not meaningful
under per-token KL). Also consider capping video tokens via the processor
(`fps` / `max_pixels`) — 14k tokens is slow and inflates KL for no clear benefit.

### 2.5 Problem D — wrong selectivity axis (the core scientific fix)

IB removes **nuisance** and keeps **signal**. "Signal" = features predictive of
`Y` / needed for cross-modal grounding. The literature says sinks are **not**
signal. So replace the sink axis with a **relevance axis**, following what works:

- **AdaVIB-style:** modulate per-token noise by an entropy/relevance score
  (low-relevance tokens get more noise/compression; high-relevance preserved).
- **VIBRA-style:** a query-aware gate (condition the bottleneck on the question).

Keep the sink machinery **only as an interpretability probe / ablation arm**
(§4, E5) to *test* whether sinks are informative on Qwen3-Omni AV — likely
confirming they are not, which is itself a citable contribution.

### 2.6 Problem E — β warmup

Fixed β from step 0 risks early posterior collapse (KL dominates before the
decoder learns to use `z`). Standard practice: linear-anneal β from 0→target
over the first K steps (or cyclic). Add `--beta-warmup-steps`.

---

## 3. Confounds to remove in code (Phase 0)

These are surgical and must land before any experiment is trustworthy.

| # | Confound | Where | Fix |
|---|---|---|---|
| 0a | **Aux heads leak the answer + are 620M params** (2× `Linear(2048→~152k)`), loss 0.21 ≫ nll 4e-4, discarded at inference | `av_model_v6.py:330-331,474-477` | Add `--aux-mode {off,answer_vocab}`, **default off**. If kept, project to the 43-word `ANSWER_VOCAB_LIST` only. Note: stop-grad does NOT fix leakage (the same `z` reaches the LLM); only removal or a separate non-spliced path does. |
| 0b | **KL summed over ~14k tokens** → β not length-invariant | `av_model_v6.py` sink KL `.sum()` | Reduce as per-token **mean**; re-derive β grid. Expose `--kl-reduction`. |
| 0c | **Sink variance-KL init artifact** | `av_model_v6.py:126-129` | Preserved tokens bypass KL (passthrough), don't get near-zero σ fought by KL. |
| 0d | **No β warmup** | `train/loop.py` | Add linear β warmup `--beta-warmup-steps`. |
| 0e | **Memorization regime** (2k samples, 1 epoch, NLL→0) | qsub pilots | Train/eval on a proper held-out split; report held-out NLL/acc, not training NLL. |

Keep each change behind a flag so the old behavior is reproducible. Commit Phase 0
as its own PR-sized unit with a smoke test (`v6_train_smoke.py`) proving β=0 +
noise-off + aux-off still equals vanilla.

---

## 4. The explainability instrument — `av_ib/eval/explain.py`

This is the heart of the thesis. Build it as one script with subcommands. Each
produces a JSON + a figure. **E1 gates everything else.**

### E1 — AV-reliance ablation (THE gate; build first)
For N held-out samples, measure ΔNLL and answer-flip rate under:
(a) `z` passthrough (normal), (b) `z` zeroed, (c) `z` mean-replaced,
(d) `z` from a **mismatched** video (cross-sample swap), (e) audio-only / video-only.
- **If zeroing/mismatching `z` barely changes NLL or answers**, the model is
  running on language priors and *no bottleneck on `z` can matter* → report this
  and pick a more AV-dependent task subset (or a harder benchmark). This single
  number decides whether the whole IB program is testable on this data.
- Reuse the splice path; add a provider mode that overwrites `z` per condition.

### E2 — KL / relevance attribution mapped to frames & time
Per-token KL is already logged (`kl_per_tok_max/p90`). Extend to dump per-token
KL and map indices back to **video frame index** and **audio time**. Show *which*
tokens the bottleneck compresses. Overlay on the frame grid. Question answered:
"does the IB compress background/nuisance, or salient grounding regions?"

### E3 — Splice drift maps
Generalize `CMIB_DRIFT_PROBE` (`splicing.py:197`) from one scalar to **per-token**
cos / relL2 (how much each token is rewritten by VIB+fusion). Correlate drift
with KL and with sink/non-sink. Question: "where does the transformation act?"

### E4 — Eval-time β rate–distortion curve (the money figure)
At **inference**, scale the bottleneck (e.g. multiply logvar / interpolate
`z = x + s·(μ−x)` for `s∈[0,1]`, and sweep an effective β) on a *trained*
checkpoint, and plot **held-out NLL and AVHBench-F1 vs compression level**. This
is the empirical IB curve: it shows the compression budget where hallucination
drops before task accuracy collapses — the central thesis evidence.

### E5 — Sink audit (test the ICLR-2025 claim on Qwen3-Omni AV)
- Re-discover massive-activation dims **causally** for this checkpoint: find dims
  with >1000× median activation; zero each and confirm the sink collapses.
  Verify whether `{985,1992}` actually hold (do **not** assume).
- Ablation contrast: mask sink tokens vs mask equal-count random tokens; measure
  ΔNLL / Δhallucination. If sinks are low-info (expected), report it — this
  *retires the sink-protection mechanism with evidence* and becomes a finding.

---

## 5. Corrected evaluation harness (Phase 2)

1. **Held-out MUSIC-AVQA split by `video_id`** (no video appears in both train
   and val) → val NLL + accuracy via `answer_parser`.
2. **AVHBench in the loop**: wire `eval/eval_avhbench_v6.py` (3 binary tasks:
   Audio-driven Video Hall., Video-driven Audio Hall., AV Matching) to run at
   each checkpoint. Report Acc/Prec/Rec/**F1** **and the Yes-bias %** (v3
   historically "predicted mostly No" — bias is a failure mode to watch).
   Consider adding AV Captioning (METEOR/CIDEr) later for completeness.
3. **Prior-only baseline**: same checkpoint with `z` ablated (E1 condition b);
   `AV-grounded accuracy = full − prior-only`. Report β effects on the
   *AV-grounded* component, not raw accuracy.
4. **Matched-param control**: a baseline with the same trainable budget (LoRA
   only, no VIB) so gains aren't just "more parameters."

---

## 6. The experiment (Phase 3) — one clean study, not 8 × 1 seed

**Pick ONE architecture** to make a clean claim. Recommendation:
`b_topk_nofusion` *modified* per §2.5 (relevance-adaptive, not sink-protecting),
**no fusion** (fusion has degraded results in every generation; isolate the IB
claim). Audio uses the relevance-adaptive VIB too.

- **β grid** (post-0b/0c, per-token KL): `{0, β1, β2, β3}` chosen so β·KL ≈
  `{0, 0.1, 0.5, 1.0}×` *held-out* NLL. Compute from a corrected `kl_study`.
- **β warmup** on; **aux off**; **≥3 seeds** per β.
- **Horizon:** train on the full (or large held-out-validated) set to a real
  stopping point, not 1 epoch on 2k.
- **Primary metric:** AVHBench F1 (+ Yes-bias). **Secondary:** held-out
  AV-grounded MUSIC-AVQA accuracy. **Diagnostic:** E4 curve per checkpoint.
- **Decision rule:** `β* = argmax AVHBench-F1` s.t. held-out accuracy drop `< ε`
  (set ε, e.g. 1 pt, in advance).

If a relevance-adaptive arm beats baseline on AVHBench-F1 at matched accuracy →
**positive result**. If not → **negative result with mechanism** (E1–E5 explain
*why*: e.g. model is prior-driven, or encoder features already minimal, or
compression removes grounding). Both are a thesis.

---

## 7. Phase 4 — scale or write up

**If positive:** add arms — (i) AdaVIB entropy-adaptive vs VIBRA query-aware;
(ii) wire `SinkSymmetricFusion` as a *fusion* arm (operationalize sinks in fusion,
not KL) and compare; (iii) **PAS** (training-free M-RoPE temporal stabilizer) as
an orthogonal *inference-time* add-on — it acts in the LLM's MHA after the splice,
so it can stack with the IB; (iv) **FINER** as a fine-grained negative-query
hallucination eval (read the PDF first — needs poppler: `apt-get install
poppler-utils` or `conda install -c conda-forge poppler`).

**If negative:** the deliverable is the mechanistic explanation (E1–E5 figures +
the sink-audit finding that contradicts the protection premise on Qwen3-Omni AV),
plus the corrected method as "what we tried and why pre-compressed AV encoder
features resist further IB." The README already frames this as defensible.

---

## 8. Execution checklist (do in this order)

- [ ] **P0** Phase-0 code fixes (0a–0e), each behind a flag; smoke test that
      β=0/noise-off/aux-off == vanilla. Commit.
- [ ] **P1a** Build `explain.py E1` (AV-reliance). Run on 100 held-out samples
      with the **current** best checkpoint (or untrained, to bound it).
      **GATE:** if ΔNLL(zero z) is negligible, escalate to me — the data isn't
      AV-dependent and the plan changes.
- [ ] **P1b** Build E5 sink-audit; re-validate `{985,1992}` causally; run the
      sink-vs-random masking contrast. Decide sink mechanism's fate with evidence.
- [ ] **P1c** Build E2/E3/E4.
- [ ] **P2** Held-out split + AVHBench-in-loop + prior-only + matched-param
      control. Re-run a corrected `kl_study` for the per-token β grid.
- [ ] **P3** One architecture, relevance-adaptive, β grid × ≥3 seeds, full
      horizon. Primary = AVHBench-F1.
- [ ] **P4** Scale (positive) or write the mechanistic negative result.

### Decision gates that require checking with Souleiman
- E1 shows the model ignores AV input → reconsider dataset/benchmark.
- E5 confirms sinks are low-info → formally retire sink-protection (big design change).
- Phase-3 β sweep shows no β beats baseline on AVHBench → commit to the
  negative-result write-up.

---

## 9. Appendix — key code pointers

- VIB primitive + residual + KL: `av_ib/model/bottleneck.py:75`
- SinkAwareVIB (sink classify `:107`, KL split `:126`, `beta_sink_ratio`):
  `av_ib/model/av_model_v6.py:104`
- Provider / splice orchestration + AdaVIB beta: `av_ib/model/av_model_v6.py:391`
- Aux heads (the 620M confound): `av_ib/model/av_model_v6.py:330,474`
- Splice + drift probe: `av_ib/backbone/splicing.py:136,197`
- NLL + frame/processor config (token count lever): `av_ib/backbone/qwen_omni.py:151,206`
- Loss composition + fixed β: `av_ib/train/loop.py:38`
- AVHBench eval (the real metric): `av_ib/eval/eval_avhbench_v6.py`
- KL decomposition study: `av_ib/eval/kl_study.py`
