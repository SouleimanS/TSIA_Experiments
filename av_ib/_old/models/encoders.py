"""Video and audio encoders, plus their Q-Former-like projectors.

Each builder returns an nn.Module with a documented forward signature.
For smoke testing, the `*_stub` variants are random-init modules with no
weight loading — they produce correctly-shaped outputs and that is all
the smoke test needs. Real encoders (EVA-CLIP for video, BEATs/Whisper
for audio) will be added as additional registry entries.

Forward contracts:

    VideoEncoder.forward(x_v) -> (B, N_v, d_v)
        x_v: raw video tensor with shape (B, T_v, H, W, 3) or
             pre-extracted features with shape (B, N_v, d_v).
             Stubs accept either by checking dim count.

    AudioEncoder.forward(x_a) -> (B, N_a, d_a)
        x_a: raw audio waveform (B, L_a) or features (B, N_a, d_a).

    Projector.forward(E) -> (B, num_tokens, d_out)
        Maps encoder output to the LLM hidden dimension and a fixed
        token count. Real version is a Q-Former; stub is a linear +
        learned token-count adapter.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class VideoEncoderStub(nn.Module):
    """Random-init video encoder. Accepts either (B,T,H,W,3) or (B,N,d)
    and emits (B, N_out, d_out). Trainable so the smoke test exercises
    backward, but frozen in real runs via freeze() helper."""

    def __init__(self, d_out: int = 768, n_tokens: int = 64):
        super().__init__()
        self.d_out = d_out
        self.n_tokens = n_tokens
        # Tiny learnable projection to make the module non-trivial under autograd.
        self.proj = nn.Linear(d_out, d_out)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 5:
            # (B, T, H, W, 3) -> just take a deterministic feature: pixel mean per frame.
            b, t, h, w, c = x.shape
            feats = x.float().mean(dim=(2, 3))          # (B, T, 3)
            # Tile/pad to d_out.
            reps = (self.d_out + c - 1) // c
            feats = feats.repeat(1, 1, reps)[..., : self.d_out]  # (B, T, d_out)
            # Adapt token count via linear interpolation along time.
            feats = nn.functional.interpolate(
                feats.transpose(1, 2), size=self.n_tokens, mode="linear", align_corners=False
            ).transpose(1, 2)                            # (B, n_tokens, d_out)
        elif x.dim() == 3:
            feats = x
        else:
            raise ValueError(f"Unexpected video shape {tuple(x.shape)}")
        return self.proj(feats)


class AudioEncoderStub(nn.Module):
    """Random-init audio encoder. Accepts (B, L) waveform or (B, N, d) features."""

    def __init__(self, d_out: int = 768, n_tokens: int = 64):
        super().__init__()
        self.d_out = d_out
        self.n_tokens = n_tokens
        self.proj = nn.Linear(d_out, d_out)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            # (B, L) waveform -> chunk into n_tokens, then project up to d_out.
            b, ell = x.shape
            chunk = ell // self.n_tokens
            if chunk == 0:
                # pad
                pad = torch.zeros(b, self.n_tokens - ell, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
                chunk = 1
            x = x[:, : chunk * self.n_tokens].reshape(b, self.n_tokens, chunk)
            # Project the chunk dim up to d_out by tiling.
            reps = (self.d_out + chunk - 1) // chunk
            feats = x.repeat(1, 1, reps)[..., : self.d_out]
        elif x.dim() == 3:
            feats = x
        else:
            raise ValueError(f"Unexpected audio shape {tuple(x.shape)}")
        return self.proj(feats)


_VIDEO_ENCODER_REGISTRY = {
    "random_stub": VideoEncoderStub,
    # 'eva_clip': lambda **kw: build_eva_clip(**kw),  # add when ready
}

_AUDIO_ENCODER_REGISTRY = {
    "random_stub": AudioEncoderStub,
    # 'beats': ..., 'whisper': ...
}


def build_video_encoder(name: str, **kwargs) -> nn.Module:
    if name not in _VIDEO_ENCODER_REGISTRY:
        raise KeyError(f"Unknown video encoder {name!r}; have {list(_VIDEO_ENCODER_REGISTRY)}")
    return _VIDEO_ENCODER_REGISTRY[name](**kwargs)


def build_audio_encoder(name: str, **kwargs) -> nn.Module:
    if name not in _AUDIO_ENCODER_REGISTRY:
        raise KeyError(f"Unknown audio encoder {name!r}; have {list(_AUDIO_ENCODER_REGISTRY)}")
    return _AUDIO_ENCODER_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Projectors
# ---------------------------------------------------------------------------

class LinearProjectorStub(nn.Module):
    """Stand-in for a Q-Former. Two ops:
       1. Resample along token axis to `num_tokens` via 1D interpolation.
       2. Linear-project channel dim from d_in to d_out.

    Order matters less than that both happen and shapes line up. Real
    Q-Former uses learned queries + cross-attention; this is here only
    so the smoke test can run without a 1B-param checkpoint."""

    def __init__(self, d_in: int, d_out: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, E: Tensor) -> Tensor:
        # E: (B, N, d_in) -> resample tokens -> linear
        E_resampled = nn.functional.interpolate(
            E.transpose(1, 2), size=self.num_tokens, mode="linear", align_corners=False
        ).transpose(1, 2)
        return self.linear(E_resampled)


_PROJECTOR_REGISTRY = {
    "linear_stub": LinearProjectorStub,
    # 'qformer': ...
}


def build_projector(name: str, d_in: int, d_out: int, num_tokens: int, **kwargs) -> nn.Module:
    if name not in _PROJECTOR_REGISTRY:
        raise KeyError(f"Unknown projector {name!r}; have {list(_PROJECTOR_REGISTRY)}")
    return _PROJECTOR_REGISTRY[name](d_in=d_in, d_out=d_out, num_tokens=num_tokens, **kwargs)


def freeze(module: nn.Module) -> nn.Module:
    """Freeze all parameters of a module (set requires_grad=False)."""
    for p in module.parameters():
        p.requires_grad = False
    return module
