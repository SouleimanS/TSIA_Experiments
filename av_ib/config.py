"""Configuration: dataclasses + YAML loader with deep-merge.

A run is specified by a YAML file that inherits from `configs/base.yaml`.
Each variant gets one YAML; CLI overrides land on top.

Why not Hydra: Hydra is fine but adds a dependency and a launcher contract.
For a project that will mostly be launched via SLURM/torchrun, a plain
YAML loader is enough and keeps the config-resolution behaviour visible
in one file.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Dataclasses. These describe the schema; YAML must conform to it.
# ---------------------------------------------------------------------------

@dataclass
class EncoderCfg:
    """Specifies one encoder. `name` selects which builder to call;
    `kwargs` is passed through. For the smoke test, `name='random_stub'`
    produces fixed-output tensors with no real model loaded."""
    name: str = "random_stub"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectorCfg:
    """Q-Former-like projector. Maps encoder output (d_in) to LLM hidden
    dimension (d_out) and a fixed number of tokens."""
    name: str = "linear_stub"
    num_tokens: int = 32
    d_in: int = 768
    d_out: int = 4096
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionCfg:
    """Fusion module. `name='identity'` concatenates Zv and Za along the
    token axis with no learned mixing — this is the no-cross-attention
    baseline. `name='mutual_cross_attention'` is the directive's Φ."""
    name: str = "identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckCfg:
    """Bottleneck module. `name='identity'` is the no-VIB baseline.
    `name='vib'` is the single post-fusion VIB. `name='per_modality_vib'`
    places one VIB per projector output (ADAVIB-style)."""
    name: str = "identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCfg:
    """LLM backbone. `name='tiny_stub'` is a 1-layer transformer for
    smoke-testing without downloading checkpoints."""
    name: str = "tiny_stub"
    hidden_size: int = 4096
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCfg:
    video_encoder: EncoderCfg = field(default_factory=EncoderCfg)
    audio_encoder: EncoderCfg = field(default_factory=EncoderCfg)
    video_projector: ProjectorCfg = field(default_factory=ProjectorCfg)
    audio_projector: ProjectorCfg = field(default_factory=ProjectorCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    bottleneck: BottleneckCfg = field(default_factory=BottleneckCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)


@dataclass
class TrainCfg:
    batch_size: int = 2
    num_steps: int = 1                # smoke test: one step
    beta: float = 0.0                  # only used when bottleneck != identity
    lr: float = 1e-4
    seed: int = 0
    device: str = "cuda"               # set to "cpu" for CPU runs
    dtype: str = "float32"             # bf16/fp16 for real runs


@dataclass
class DataCfg:
    """Smoke test uses synthetic batches. Real runs will point at AVQA etc."""
    name: str = "dummy"
    # synthetic-batch shape; ignored for real datasets
    video_tokens: int = 32             # Nv
    audio_tokens: int = 32             # Na
    answer_len: int = 8                # Ly
    vocab_size: int = 32000


@dataclass
class RunCfg:
    """Top-level config. Every YAML loads into one of these."""
    name: str = "unnamed"
    variant: str = "v1_no_fusion_no_vib"   # one of the six in the plan
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)


# ---------------------------------------------------------------------------
# YAML I/O. The loader handles 'inherits: <path>' for composition.
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. Lists and scalars are
    replaced wholesale; dicts are merged key-by-key."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_inherits(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load a YAML, following an `inherits:` key (single parent) recursively.
    Detects cycles."""
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Cyclic inherits chain at {path}")
    _seen.add(path)

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    parent_rel = data.pop("inherits", None)
    if parent_rel is None:
        return data

    parent_path = (path.parent / parent_rel).resolve()
    parent = _load_yaml_with_inherits(parent_path, _seen)
    return _deep_merge(parent, data)


def _dict_to_runcfg(d: dict) -> RunCfg:
    """Build a typed RunCfg from a plain dict by walking the schema.
    Unknown keys raise — early failure beats silent typos."""
    def build(cls, payload: dict | None):
        payload = payload or {}
        if not hasattr(cls, "__dataclass_fields__"):
            return payload
        fields_ = cls.__dataclass_fields__
        unknown = set(payload) - set(fields_)
        if unknown:
            raise ValueError(f"Unknown keys for {cls.__name__}: {unknown}")
        kwargs = {}
        for fname, finfo in fields_.items():
            if fname not in payload:
                continue
            ftype = finfo.type
            # Resolve string annotations to real types via globals of this module.
            if isinstance(ftype, str):
                ftype = eval(ftype, globals())   # noqa: S307
            if hasattr(ftype, "__dataclass_fields__"):
                kwargs[fname] = build(ftype, payload[fname])
            else:
                kwargs[fname] = payload[fname]
        return cls(**kwargs)
    return build(RunCfg, d)


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunCfg:
    """Load a YAML config file. `overrides` is a list of `dotted.key=value`
    strings (e.g. `train.batch_size=4`); they are applied last.

    Override values are parsed as YAML scalars, so `train.beta=1e-3` becomes
    a float, `train.device=cpu` stays a string."""
    raw = _load_yaml_with_inherits(Path(path))
    for ov in (overrides or []):
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        key, _, val_str = ov.partition("=")
        val = yaml.safe_load(val_str)
        # Walk dotted key, creating intermediate dicts.
        d = raw
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return _dict_to_runcfg(raw)


def runcfg_to_dict(cfg: RunCfg) -> dict:
    return asdict(cfg)
