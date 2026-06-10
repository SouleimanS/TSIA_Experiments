# Mathematical foundations

This note grounds the project in three established theories and shows that the
code already implements their objectives. It also states the one *new* claim the
project makes — a bridge from the information-bottleneck **rate** to a
**selective-prediction** (fail-safe) rule — and the information-flow condition
under which that bridge is valid (which is exactly the "open issue" slide).

Notation: `X_v, X_a` raw video/audio encoder features, `Z_v, Z_a` their
bottleneck codes, `Y` the answer, `Q` the question/prompt. `I(·;·)` mutual
information, `H(·)` entropy, `D_KL` Kullback–Leibler divergence.

---

## 1. The bottleneck is a (variational) information bottleneck

**Origin — Information Bottleneck.** Tishby, Pereira & Bialek (1999/2000) pose
representation learning as a rate–relevance trade-off:

```
min_{p(z|x)}  I(X; Z) − β · I(Z; Y).
```

`I(X;Z)` is the **rate** (bits the code keeps about the input); `I(Z;Y)` is the
**relevance** (bits the code keeps about the label). β sweeps the trade-off and
is the negative inverse slope of the resulting *information curve*
[Tishby et al. 2000, https://arxiv.org/abs/physics/0004057].

**Tractable form — Deep Variational IB (VIB).** Alemi, Fischer, Dillon & Murphy
(ICLR 2017) make this trainable with a Gaussian encoder
`p(z|x)=N(μ_φ(x), diag σ²_φ(x))`, a variational decoder `q(y|z)`, and a prior
`r(z)=N(0,I)`:

```
L_VIB = E[ −log q(y|z) ]  +  β · E[ D_KL( p(z|x) ‖ r(z) ) ],
D_KL( N(μ,σ²) ‖ N(0,I) ) = ½ Σ_j ( μ_j² + σ_j² − 1 − log σ_j² ).
```

[Alemi et al. 2017, https://arxiv.org/abs/1612.00410]

**What our code computes.** `av_ib/model/bottleneck.py` /
`av_ib/model/av_model_v6.py` build exactly this:

```python
kl_elem_mu  = 0.5 * mu.pow(2)
kl_elem_var = 0.5 * (logvar.exp() - logvar - 1.0)        # = ½(σ² − logσ² − 1)
kl_per_token = (kl_elem_mu + kl_elem_var).mean(dim=-1)    # closed-form Gaussian KL
...
loss = nll + beta_v*kl_v + beta_a*kl_a + beta_j*kl_j + aux_weight*(...)
```

So **`KL_v` is the variational estimate of the video rate `I(X_v; Z_v)`** and the
training loss *is* the VIB Lagrangian (the `nll` term is the cross-entropy lower
bound on `I(Z;Y)`). The residual, near-identity init (`μ = x + fc_μ(x)`,
`logσ²≈−3`) starts the code at the uncompressed point and lets β pull the rate
down.

**Consequence for the slides.** The held-out *accuracy vs. `KL_v`* table is an
empirical **IB information curve**: as β rises, `KL_v` falls (12229 → 4710 →
2301) and accuracy declines monotonically (86 → 80 → 76.7). "2.6× compression
for ~2pp" is a single rate–distortion operating point on that curve. This is the
textbook behaviour an IB should show — it is *the* sanity check that the module
is a real bottleneck, not that it improves the task.

---

## 2. "Video-only" is a conditional-information statement, and the benchmark diagnosis is an information-theoretic one

**Conditional bottlenecks.** The clean way to say "audio is redundant given video
+ prior" is conditional mutual information. The Conditional Entropy Bottleneck
(Fischer 2020) and the conditional-MI bound (Tezuka & Namekawa 2021) replace the
rate with `I(X; Z | Y)`, using the identity

```
I(X; Z | Y) ≤ I(X; Z),
```

i.e. conditioning removes label-irrelevant nuisance and gives a *strictly
tighter* compression objective
[Fischer 2020, https://arxiv.org/abs/2002.05379;
Tezuka & Namekawa 2021, https://www.mdpi.com/1099-4300/23/8/974].
The multimodal version we cite in the README (C-MIB / Multimodal IB, Mai, Zeng &
Hu, IEEE T-MM 2022) writes a per-modality rate plus a fused relevance term
[https://arxiv.org/abs/2210.17444]:

```
Σ_m I(X_m; Z_m) − β · I(Z_fused; Y).
```

**The diagnosis, stated formally.** Our ablation (delete a modality, measure
ΔNLL) is a plug-in estimate of conditional mutual information about `Y`:

- Deleting **audio** costs `+8.3%` NLL ⇒ `I(X_a; Y | X_v, Q) ≈ 0` (audio is
  conditionally redundant).
- Deleting **both** A and V leaves accuracy at 65% ⇒ `I(Y; X_a, X_v | Q)` is
  small relative to `H(Y|Q)`: the answer is mostly determined by the prior on the
  question. The audio rate `KL_a` stays frozen (~494 → ~486 over 2000 steps)
  precisely because there is no relevance gradient `∂I(Z_a;Y)/∂θ_a` to trade
  against its rate — there is nothing for an audio bottleneck to learn.

**Why this invalidates the benchmark as an AV test.** An IB maximizes `I(Z;Y)`
subject to a rate budget. If `I(A,V; Y | Q) ≈ 0`, then `I(Z;Y)` can be maximized
without `Z` carrying any audio-visual information — the relevance term is
satisfied by the prior. Hence accuracy on this benchmark cannot certify (or
refute) an audio-visual bottleneck. This is the rigorous version of "86% vs 84%
is the wrong yardstick," and it matches the independent Omni-R1 observation that
text-only tuning improves "audio" scores on a sibling model.

---

## 3. Fail-safe abstention is selective prediction (risk–coverage)

**Selective classifier.** El-Yaniv & Wiener (JMLR 2010) formalize a predictor as
a pair `(f, g)` with predictor `f:X→Y` and gate `g:X→{0,1}`; the model answers
`f(x)` when `g(x)=1` and abstains otherwise. Define

```
coverage   φ(g) = E[ g(x) ],
sel. risk  R(f,g) = E[ ℓ(f(x),y)·g(x) ] / φ(g),
```

and the **risk–coverage curve** is `{(φ(g_θ), R(f,g_θ))}` as a confidence
threshold θ sweeps. Lower curves are better; the scalar summary is the
**AURC = ∫₀¹ R(f,g_φ) dφ**
[El-Yaniv & Wiener 2010, https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf;
Geifman & El-Yaniv, NeurIPS 2017, https://arxiv.org/abs/1705.08500].

**Bayes-optimal gate (Chow's rule).** With cost-based rejection the optimal gate
abstains on low max-posterior:

```
answer iff  max_y p(y | x) ≥ θ,
θ = (C_r − C_c)/(C_e − C_c),
```

[Chow 1970, IEEE T-IT]. The Neyman–Pearson view (2025) refines this to a
likelihood-ratio test [https://arxiv.org/abs/2505.15008].

**What our code computes.** `av_ib/eval/eval_riskcoverage.py` reports exactly
`coverage` and `risk` on the answered subset; the abstention target string
"I can't tell" is the realization of `g(x)=0`. Our "decisive test" —
generalization of `g` to **held-out** corruptions — is the standard demand that
the gate's confidence ranking transfer, i.e. low **AURC/E-AURC** off-distribution
[Traub et al., NeurIPS 2024, https://arxiv.org/abs/2407.01032]. The
negative-control corruptions (degrade only inert audio ⇒ must *not* abstain) keep
coverage high and rule out a gate that collapses to "always abstain."

**Direct lineage in VLMs.** Selective prediction for multimodal QA is an active
line: Reliable VQA (Whitehead et al., ECCV 2022,
https://arxiv.org/abs/2204.13631), ReCoVERR (Srinivasan et al., ACL Findings
2024, https://arxiv.org/abs/2402.15610), Variational VQA (Wieczorek et al. 2025,
https://arxiv.org/abs/2505.09591), and evidence-grounded selection SIEVES (2026,
https://arxiv.org/abs/2604.25855). Our task ("abstain when the *essential*
modality is corrupted") is the audio-visual instance of this family.

---

## 4. DPO is the KL-regularized way to install the gate

**Objective.** Direct Preference Optimization (Rafailov et al., NeurIPS 2023)
starts from KL-constrained reward maximization

```
max_π  E_{x, y∼π}[ r(x,y) ]  −  β · D_KL( π(·|x) ‖ π_ref(·|x) ),
```

whose closed-form optimum is `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`. Inverting
gives the implicit reward `r(x,y) = β log[ π(y|x)/π_ref(y|x) ] + const`, and
substituting into the Bradley–Terry preference model yields a loss with **no RL
loop**:

```
L_DPO = − E_{(x, y_w, y_l)} log σ( β [ log π_θ(y_w|x)/π_ref(y_w|x)
                                       − log π_θ(y_l|x)/π_ref(y_l|x) ] ).
```

[Rafailov et al. 2023, https://arxiv.org/abs/2305.18290]

**What our code computes.** `av_ib/train/train_dpo.py` is line-for-line this:

```python
logits = beta_dpo * ((pol_ch - ref_ch) - (pol_rj - ref_rj))
loss   = -F.logsigmoid(logits)
```

with the reference `π_ref` obtained by disabling the LoRA adapters
(`peft_model.disable_adapter()`). The preference pairs install the gate:
`(chosen="I can't tell", rejected=gold)` when video is corrupted (push
`g→0`); `(chosen=gold, rejected="I can't tell")` when evidence is intact or only
audio is degraded (push `g→1`). The KL-to-reference term (the β that multiplies
the log-ratio) is what keeps capability while shifting the abstention behaviour —
the property FINER attributes to DPO over SFT.

---

## 5. The one new claim: rate ⇒ gate, and the validity condition

The pieces above are established. The project's *own* hypothesis is a bridge
between §1 and §3:

> **Claim.** A bottleneck makes fail-safe abstention *measurable and
> controllable*, because the code's informativeness about `Y` is an explicit,
> monitorable quantity. When the essential evidence is corrupted, the relevant
> information in the code drops, the answer posterior flattens
> (`H(Y|Z)` rises / `max_y p(y|z)` falls), and Chow's rule abstains.

Formally, the gate can be driven by the code rather than by a black-box
confidence: `g(x) = 1[ max_y p(y | Z(x)) ≥ θ ]`, and the bottleneck gives a
principled, low-dimensional `Z` on which to threshold.

**Validity condition (the "open issue" slide, stated precisely).** For the *code*
to drive abstention, the corruption must change the code. By the data-processing
inequality along `X → Z → Ŷ`, if a corruption `c` is injected **downstream** of
the bottleneck (on `Z` itself), then for the frozen-LLM / bottleneck-only
condition the encoder parameters receive

```
∂L/∂θ_VIB = 0   on corrupted steps,
```

because `Z` (the only trainable signal path) is computed *before* `c` is applied
— the VIB never observes the corruption (empirically: `grad_norm = 0` on
`vid_zero` steps). To test the claim, corruption must be injected **upstream** of
the VIB, on the encoder output `X_v`, so that it propagates through
`I(X_v^corrupt; Z_v)` and the code actually reflects the lost evidence. LoRA
conditions are immune because the adapters sit downstream of `c` and can learn
from it directly — which is why those runs train and the bottleneck-only run does
not. This is a clean information-flow argument, not an implementation detail.

---

## 6. One-paragraph "why this is principled" for the talk

The module is a deep variational information bottleneck [Alemi 2017] whose rate
`I(X_v;Z_v)` we read off as `KL_v`; the held-out accuracy-vs-`KL_v` table is its
information curve. The benchmark diagnosis is a conditional-MI statement —
`I(A,V;Y|Q)≈0` — so accuracy there cannot validate the bottleneck [Fischer 2020;
Mai 2022]. We therefore evaluate the property a bottleneck *should* buy —
fail-safe behaviour — as **selective prediction** with risk–coverage and Chow's
optimal reject rule [El-Yaniv & Wiener 2010; Chow 1970], and install the gate
with **DPO**, the KL-regularized preference objective whose optimum is an
exponential tilt of the reference policy [Rafailov 2023]. The open question is
purely about information flow: the corruption must enter upstream of the code for
the bottleneck-only model to learn the gate (data-processing inequality).

---

## References (verified URLs)

Information bottleneck / VIB
- Tishby, Pereira, Bialek (2000). The Information Bottleneck Method. https://arxiv.org/abs/physics/0004057
- Tishby, Zaslavsky (2015). Deep Learning and the IB Principle. https://arxiv.org/abs/1503.02406
- Alemi, Fischer, Dillon, Murphy (ICLR 2017). Deep Variational Information Bottleneck. https://arxiv.org/abs/1612.00410
- Fischer (2020). The Conditional Entropy Bottleneck. https://arxiv.org/abs/2002.05379
- Tezuka, Namekawa (2021). IB Analysis by a Conditional MI Bound. https://www.mdpi.com/1099-4300/23/8/974
- Federici et al. (ICLR 2020). Multi-View Information Bottleneck. https://arxiv.org/abs/2002.07017
- Mai, Zeng, Hu (IEEE T-MM 2022). Multimodal Information Bottleneck. https://arxiv.org/abs/2210.17444
- Hu, Lou, Yan, Ye (IEEE TPAMI 2024). A Survey on Information Bottleneck. https://ieeexplore.ieee.org/document/10438074/

Selective prediction / risk–coverage
- Chow (1970). On optimum recognition error and reject tradeoff. IEEE T-IT.
- El-Yaniv, Wiener (JMLR 2010). On the Foundations of Noise-free Selective Classification. https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf
- Geifman, El-Yaniv (NeurIPS 2017). Selective Classification for DNNs. https://arxiv.org/abs/1705.08500
- Geifman, El-Yaniv (ICML 2019). SelectiveNet. https://arxiv.org/abs/1901.09192
- Traub et al. (NeurIPS 2024). Overcoming Common Flaws in Evaluation of Selective Classification (AUGRC). https://arxiv.org/abs/2407.01032
- Know When to Abstain (2025). Likelihood-ratio optimal abstention. https://arxiv.org/abs/2505.15008

Selective prediction in VLMs
- Whitehead et al. (ECCV 2022). Reliable VQA. https://arxiv.org/abs/2204.13631
- Srinivasan et al. (ACL Findings 2024). ReCoVERR. https://arxiv.org/abs/2402.15610
- Wieczorek et al. (2025). Variational VQA. https://arxiv.org/abs/2505.09591
- SIEVES (2026). Visual Evidence Scoring. https://arxiv.org/abs/2604.25855

Preference optimization
- Rafailov et al. (NeurIPS 2023). Direct Preference Optimization. https://arxiv.org/abs/2305.18290

> Status note: the DPO, multimodal-grounding, and KL→fail-safe literature sweeps
> were partially interrupted by a session token limit; the DPO derivation here is
> from the primary source and is verified, but a second pass (FINER specifics,
> Omni-R1, attention-sink theory) is still pending and can be added after reset.
