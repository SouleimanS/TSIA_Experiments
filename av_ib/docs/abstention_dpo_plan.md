# Parallel workstream: DPO abstention (FINER-inspired)

This runs alongside the agreed Stage-0 / abstain-label plan (jobs `stage0_c0`,
`abstain_labels`). It does **not** change those jobs. It adds a *preference-based*
training path to compare against the SFT abstain-loss path, motivated by two papers.

## What the two papers give us

### Omni-R1 (MIT, 2505.09439) — corroborates Asset A
On the sibling model (Qwen2.5-Omni) they show that (a) text-only fine-tuning
(audio dropped) *improves* audio-benchmark accuracy and (b) audio-withheld
inference shows most of the gain is better text reasoning, not better audio use.
This is independent evidence for our finding that the modality is largely inert
and the benchmark is solvable from the prior. We cite it as external validation;
our contribution is the *causal, per-channel* version (E1 ablations, KL_a frozen
~477, video_only costs only +4.4% NLL) **plus a remedy** (fail-safe abstention).
Omni-R1 does NOT propose a remedy — no scoop on our direction.

### FINER (CVPR 2026) — method input for the remedy
MLLMs hallucinate "yes" under fine-grained negative queries (asking about content
not present). FINER-Tuning fixes this with **DPO** on preference pairs and reports
DPO > SFT for suppressing credulity *while preserving* general capability.

This maps directly onto our failure: T2 became *more credulous* on AVHBench
Video-driven Audio Hallucination (-8.3pp, yes_pct 38->57%). Evidence-removal
corruptions (vid_zero/mean) are the AV analog of FINER's "non-existent content".

## The DPO formulation (this workstream)

Preference pairs are derived **entirely from the existing miner output**
(`runs/abstain_labels.json`) — no schema change, no re-run needed:

| pair type        | source items                       | corruption | chosen        | rejected      |
|------------------|------------------------------------|------------|---------------|---------------|
| abstain          | id_correct & vid-flip & not aud-flip| vid_zero/mean | "I can't tell" | gold answer |
| capability anchor| id_correct                          | identity   | gold answer   | "I can't tell"|
| negative control | id_correct & aud-flip               | aud_zero/mean | gold answer | "I can't tell"|

Rationale per FINER: when essential evidence (video) is removed, *prefer refusal
over a confident assertion* even if the prior would have been right. When evidence
is intact (identity) or only the non-essential modality (audio) is degraded,
*prefer answering over abstaining* — this is what keeps coverage high and stops
the model from collapsing into "I can't tell" everywhere.

Loss (standard DPO, shared frozen base as reference via PEFT `disable_adapter`):

    r(x,y) = logp_policy(y|x) - logp_ref(y|x)
    L = -logsigmoid( beta_dpo * ( r(x, chosen) - r(x, rejected) ) )

Reference = LoRA adapters disabled; VIB runs under no_grad with current weights
(tiny shared module — acceptable anchor, same as TRL shared-base DPO).

## Conditions (mirror the SFT plan, swap loss)
- **C1-DPO**: vanilla + LoRA + DPO   (no bottleneck)
- **C2a-DPO**: b_video_only VIB-only + DPO  (frozen LLM, bottleneck only)
- **C2b-DPO**: b_video_only + LoRA + DPO

## The test that decides
Evaluate C1-DPO vs C2a-DPO vs C2b-DPO with `eval_riskcoverage.py` on **held-out**
corruptions (vid_noise, vid_scale — never seen in training) and the **negative
control** (aud_zero/mean). Then transfer the winner to AVHBench Video-driven
Audio Hallucination (the -8.3pp task).

Pre-committed reads (unchanged from agreed plan):
- C2 >> C1 on held-out corruptions  -> bottleneck buys fail-safe generalization. Ship.
- C2 ~= C1                           -> bottleneck not needed; DPO alone fixes it. Ship Asset A + simpler remedy.
- all ~= C0 (Stage-0 baseline)       -> drop direction; ship Asset A only.

## Gating
This workstream is *built and committed now* (no GPU). It is **submitted only
after**: (1) Stage-0 shows the abstain prompt alone does NOT already fail-safe,
and (2) the miner yields enough contrast (vid-flip, not aud-flip) pairs to train.
If Stage-0 already fail-safes, we skip training entirely and ship Asset A.
