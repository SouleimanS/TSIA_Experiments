"""Bottleneck modules. Each takes a representation and returns
(T, kl_per_example), where T is the bottlenecked representation passed
downstream and kl_per_example is a (B,) tensor of KL contributions.

Variants:
    IdentityBottleneck -- pass-through; KL is zero. Used for variant 1.

    VIB -- variational information bottleneck (eqs. 5-6 of the whitepaper).
        Per-token MLP produces (mu, logvar); samples T = mu + sigma * eps;
        KL against N(0, I) is computed in closed form per token then summed
        across the token dimension to give one KL per example.

    PerModalityVIB -- ADAVIB-style. Two independent VIBs applied to Zv and
        Za before fusion. Useful for the section-4 ablation 'is fused-stage
        bottlenecking necessary?'

Contracts:
    VIB.forward(Z)         -> (T, kl)        where Z, T: (B, N, d), kl: (B,)
    PerModalityVIB.forward -> not used here; PerModalityVIB lives upstream
                              of fusion and is invoked separately on Zv, Za.

The VIB outputs (mu, logvar) are stored on the module after each forward
pass via `last_mu` and `last_logvar` attributes. This is *only* for
diagnostics (e.g. posterior-entropy probes from the experimental plan,
step 8). The training loss does not read them; it uses the kl tensor
returned from forward.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor


def _gaussian_kl_to_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL(N(mu, sigma^2) || N(0, I)) per element, summed over feature dim.

    Standard closed form:
        KL = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )

    `logvar` is log(sigma^2). Summed over the last (feature) dim only;
    token and batch dims are preserved for the caller to reduce as
    needed."""
    return 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar).sum(dim=-1)


class IdentityBottleneck(nn.Module):
    """No bottleneck. Returns input unchanged and zero KL."""

    def __init__(self):
        super().__init__()
        self.last_mu = None
        self.last_logvar = None

    def forward(self, Z: Tensor) -> tuple[Tensor, Tensor]:
        kl = torch.zeros(Z.size(0), device=Z.device, dtype=Z.dtype)
        return Z, kl


class VIB(nn.Module):
    """Variational information bottleneck applied per token.

    Architecture (Figure 3 of the whitepaper):
        Linear d -> d_h, GELU, Linear d_h -> 2*d, split into (mu, logvar).
    Sampling:
        T = mu + sigma * eps, eps ~ N(0, I), with sigma = exp(0.5*logvar).
    """

    def __init__(self, d: int, d_h: int = 1024):
        super().__init__()
        self.d = d
        self.d_h = d_h
        self.net = nn.Sequential(
            nn.Linear(d, d_h),
            nn.GELU(),
            nn.Linear(d_h, 2 * d),
        )
        # logvar clamp to prevent numerical issues during early training.
        # Wide enough not to bind in normal use; narrow enough to catch NaN gradients.
        self.logvar_min = -10.0
        self.logvar_max = 10.0
        self.last_mu = None
        self.last_logvar = None

    def forward(self, Z: Tensor) -> tuple[Tensor, Tensor]:
        # Z: (B, N, d)
        params = self.net(Z)                               # (B, N, 2d)
        mu, logvar = params.chunk(2, dim=-1)               # each (B, N, d)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)

        if self.training:
            eps = torch.randn_like(mu)
            sigma = (0.5 * logvar).exp()
            T = mu + sigma * eps
        else:
            # At eval time use the mean. This is the standard VIB choice;
            # it gives deterministic outputs for fair comparison across runs.
            T = mu

        # KL per token (B, N), then sum over tokens -> per-example KL (B,).
        kl_per_token = _gaussian_kl_to_standard_normal(mu, logvar)
        kl = kl_per_token.sum(dim=-1)

        # Stash for diagnostics. Detached to avoid holding the graph.
        self.last_mu = mu.detach()
        self.last_logvar = logvar.detach()
        return T, kl


class PerModalityVIB(nn.Module):
    """Two independent VIBs, one per modality. Applied to Zv and Za
    *before* fusion. Used for the ADAVIB-style ablation in the
    experimental plan (step 7).

    Note: this module is invoked differently in the forward graph -- the
    av_model orchestrator calls it on each projector output separately,
    not on the fused representation. See av_model.AVModel for placement.
    """

    def __init__(self, d: int, d_h: int = 1024):
        super().__init__()
        self.vib_v = VIB(d=d, d_h=d_h)
        self.vib_a = VIB(d=d, d_h=d_h)

    def forward(self, Zv: Tensor, Za: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        Tv, kl_v = self.vib_v(Zv)
        Ta, kl_a = self.vib_a(Za)
        return Tv, Ta, kl_v + kl_a


_BOTTLENECK_REGISTRY = {
    "identity": lambda d, **kw: IdentityBottleneck(),
    "vib": lambda d, **kw: VIB(d=d, **kw),
    "per_modality_vib": lambda d, **kw: PerModalityVIB(d=d, **kw),
}


def build_bottleneck(name: str, d: int, **kwargs) -> nn.Module:
    if name not in _BOTTLENECK_REGISTRY:
        raise KeyError(f"Unknown bottleneck {name!r}; have {list(_BOTTLENECK_REGISTRY)}")
    return _BOTTLENECK_REGISTRY[name](d=d, **kwargs)
