# av_ib

Scaffolding for the audio-visual Information Bottleneck research direction.
See `Whitepaper.pdf` for motivation, objective, and architecture.

## Repo layout

```
av_ib/
├── configs/                 # one YAML per ablation variant
│   ├── base.yaml            # shared defaults
│   └── v1_no_fusion_no_vib.yaml
├── av_ib/
│   ├── config.py            # YAML loader + dataclasses + CLI overrides
│   ├── models/
│   │   ├── encoders.py      # video + audio encoders, projectors (registry)
│   │   ├── fusion.py        # Identity, MutualCrossAttention
│   │   ├── bottleneck.py    # Identity, VIB, PerModalityVIB
│   │   ├── llm.py           # TinyLLMStub, slot for HF backbones
│   │   └── av_model.py      # assembles the variants
│   ├── data/dummy.py        # synthetic batches for smoke testing
│   └── training/
│       ├── loss.py          # NLL + beta * KL
│       └── train_step.py
├── scripts/smoke_test.py    # entry point for this milestone
└── tests/test_shapes.py     # shape-contract tests
```

## The six variants

Each is a YAML in `configs/` that inherits from `base.yaml`:

| Variant | Fusion | Bottleneck | Purpose |
|---|---|---|---|
| v1 no-fusion / no-VIB | Identity | Identity | sanity baseline |
| v2 cross-attention only | Φ | Identity | does fusion capacity alone help? |
| v3 VIB only | Identity | VIB | does compression alone help? |
| v4 Φ + VIB (proposal) | Φ | VIB | the directive's main hypothesis |
| v5 separate VIBs only | Identity | PerModalityVIB | ADAVIB-style ablation |
| v6 separate VIBs + fused VIB | Identity | PerModalityVIB + VIB | compress per-modality and joint |

Only v1's YAML is written. v2-v6 are single-file additions.

## Running

```bash
# default: GPU
python scripts/smoke_test.py --config configs/v1_no_fusion_no_vib.yaml

# CPU
python scripts/smoke_test.py --config configs/v1_no_fusion_no_vib.yaml train.device=cpu

# override anything
python scripts/smoke_test.py --config configs/v1_no_fusion_no_vib.yaml \
    train.batch_size=4 train.num_steps=3 train.beta=1e-3
```

## How to add a real encoder / LLM

The registries in `encoders.py`, `fusion.py`, `bottleneck.py`, and `llm.py`
take a string name. Adding a new backend is one entry per registry and a
factory function. Example for EVA-CLIP:

```python
# in encoders.py
_VIDEO_ENCODER_REGISTRY["eva_clip"] = build_eva_clip
```

Then in the config:

```yaml
model:
  video_encoder:
    name: eva_clip
    kwargs:
      checkpoint_path: /path/to/eva_clip.pt
```

## What 'smoke test' verifies for v1

- Config loads and resolves inheritance correctly.
- Trainable-parameter breakdown is 0 across all modules (v1 has no learned parts).
- Forward pass from raw video/audio through frozen encoders, frozen
  projectors, identity fusion, identity bottleneck, frozen LLM produces
  a finite cross-entropy loss on dummy answer labels.
- KL term is zero everywhere.
- Backward runs without error (no gradient sinks, but the call is exercised).
```
