# Prompt-conditioned modality gating

A modality gate on top of the existing C-MIB (video + audio bottleneck)
pipeline. The gate reads the **prompt** and predicts how much each modality
matters, then **enhances** the injected modality tokens by scaling their
embeddings with that probability before they reach the LLM.

## Mechanism

```
prompt text ──► mean-pool frozen embeddings ──► MLP ──► softmax
                                                          │
                                          p = (p_video, p_audio, p_text)
                                                          │
   z_video  ◄── (M · p_video) · z_video   ◄──────────────┤
   z_audio  ◄── (M · p_audio) · z_audio   ◄──────────────┘
                  │
                  └─► spliced into the LLM's inputs_embeds (CMIBSplicer)
```

- **`p`** is a proper distribution over `{video, audio, text}`. The `text`
  entry is a "neither" sink: it lets the gate move probability mass *off* the
  AV streams when the prompt is self-contained.
- **`M · p`** (M = number of modalities) centres the scale at the uniform
  distribution. Uniform `p` ⇒ scale 1.0 ⇒ identity. The gate's final layer is
  zero-initialised, so **an untrained gate is exactly the ungated baseline**
  and training only *departs* from it where it helps.
- **No extra loss.** The gate trains end-to-end through the answer NLL:
  scaling the spliced embeddings changes what the LLM sees, which changes the
  answer likelihood, which back-propagates into the MLP. The 30B backbone is
  frozen; only the gate (+ existing LoRA + bottleneck) gets gradients.

## Files

| file        | what                                                       |
|-------------|------------------------------------------------------------|
| `gate.py`   | `ModalityGate` — the MLP (zero-init, fp32).                |
| `model.py`  | `AVModelGated` — `AVModelV6` + gate; overrides the provider, `forward_train`, `forward_generate`, diagnostics. |
| `train.py`  | training driver (reuses `av_ib.train.loop.run_training`).  |

## Train

```bash
qsub scripts/modality_gate/train_gate.qsub
```

or directly:

```bash
python -m av_ib.modality_gate.train \
    --dataset avqa \
    --ann-path .../train_qa.json --video-root .../videos \
    --variant b_topk_nofusion \
    --num-steps 2000 --beta-v 7e-6 --beta-a 7e-6 \
    --log-path runs/gate/gate.jsonl --ckpt-path runs/gate/gate_final.pt
```

The training log records `p_video / p_audio / p_text` per step — watch these
drift away from the initial 0.33/0.33/0.33 as the gate learns which modality
each question type needs.

## Notes

- **Batch size 1.** Like the rest of the pipeline (see `CMIBSplicer`), the gate
  assumes one sample per forward.
- **Ablation.** `--gate-raw` scales by raw `p` (always shrinks, never the
  identity) instead of the centred `M·p`, to test whether centring matters.
- The checkpoint is a `trainable_state` dict (LoRA + bottleneck + gate); load
  it into `AVModelGated`, not `AVModelV6`.
