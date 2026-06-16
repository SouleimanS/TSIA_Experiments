"""Encoder -> Variational Information Bottleneck -> Decoder for video clips.

The objective is the Deep Variational Information Bottleneck (Alemi et al.,
2017), specialised to reconstruction (the "relevant variable" Y is the input X
itself, so the model is an autoencoder):

    L(beta) = E_q[ -log p(x | z) ]  +  beta * KL( q(z | x) || p(z) )
              \-----------------/        \-------------------------/
                  distortion D                    rate R
              (reconstruction error)     (an upper bound on I(Z; X))

p(z) is a standard normal prior. The KL term is a tractable *upper bound* on the
mutual information I(Z; X) -- the number of nats the latent spends describing the
input. beta trades the two off:

    beta -> 0 : rate is free, the latent keeps everything, near-perfect decode.
    beta large: rate is expensive, the latent keeps only the most useful bits,
                reconstruction degrades gracefully (noise/detail dropped first).

Sweeping beta therefore traces a rate-distortion curve -- the IB curve.

Latent design: a single global vector z in R^d summarises the *whole clip*. With
a per-dimension KL we can count "active units" (dimensions the model actually
uses); watching that count fall as beta grows is the clearest single picture of
the bottleneck at work.

Canonical tensor layout everywhere: (B, C, T, H, W), matching nn.Conv3d.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    C: int = 1          # channels
    T: int = 8          # frames
    H: int = 32
    W: int = 32
    latent_dim: int = 16
    width: int = 32     # base channel width of the conv stacks


class VIB(nn.Module):
    """Variational Information Bottleneck head.

    Maps a deterministic feature vector to a Gaussian posterior q(z|x), samples z
    with the reparameterisation trick, and reports the per-sample KL to a
    standard-normal prior (the "rate").
    """

    def __init__(self, in_dim: int, latent_dim: int, logvar_clamp: float = 10.0):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        self.logvar_clamp = logvar_clamp

    def forward(self, h: torch.Tensor):
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-self.logvar_clamp, self.logvar_clamp)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu  # deterministic at eval: use the posterior mean
        # Per-dimension KL( N(mu, sigma^2) || N(0, 1) ), shape (B, latent_dim).
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        return z, mu, logvar, kl_per_dim


class _Encoder(nn.Module):
    """3D-conv encoder: (B, C, T, H, W) -> feature vector (B, feat_dim)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        w = cfg.width
        # Downsample H,W by 8x over three blocks; downsample T by 4x.
        self.net = nn.Sequential(
            nn.Conv3d(cfg.C, w, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv3d(w, 2 * w, 3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, 2 * w), nn.SiLU(),
            nn.Conv3d(2 * w, 2 * w, 3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, 2 * w), nn.SiLU(),
        )
        # Infer the flattened feature size with a dry run so the model adapts to
        # whatever (C, T, H, W) the config asks for.
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.C, cfg.T, cfg.H, cfg.W)
            feat = self.net(dummy)
        self.feat_shape = feat.shape[1:]            # (C', T', H', W')
        self.feat_dim = int(feat.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class _Decoder(nn.Module):
    """Mirror of the encoder: latent z (B, d) -> reconstruction (B, C, T, H, W)."""

    def __init__(self, cfg: ModelConfig, feat_shape):
        super().__init__()
        self.feat_shape = feat_shape
        w = cfg.width
        self.fc = nn.Linear(cfg.latent_dim, int(torch.tensor(feat_shape).prod()))
        self.net = nn.Sequential(
            nn.ConvTranspose3d(2 * w, 2 * w, 3, stride=(2, 2, 2),
                               padding=1, output_padding=1),
            nn.GroupNorm(8, 2 * w), nn.SiLU(),
            nn.ConvTranspose3d(2 * w, w, 3, stride=(2, 2, 2),
                               padding=1, output_padding=1),
            nn.GroupNorm(8, w), nn.SiLU(),
            nn.ConvTranspose3d(w, cfg.C, 3, stride=(1, 2, 2),
                               padding=1, output_padding=(0, 1, 1)),
        )
        self.out_size = (cfg.C, cfg.T, cfg.H, cfg.W)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.shape[0], *self.feat_shape)
        x = self.net(h)
        # Guard against off-by-one from strided (de)convolution on odd sizes.
        if x.shape[2:] != self.out_size[1:]:
            x = F.interpolate(x, size=self.out_size[1:], mode="trilinear",
                              align_corners=False)
        return torch.sigmoid(x)


class VideoIBAutoencoder(nn.Module):
    """Full encode -> VIB -> decode model.

    forward(x) returns (x_hat, kl_per_dim). Loss assembly lives in train.py so
    the rate/distortion trade-off (the choice of beta) is explicit and easy to
    sweep.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = _Encoder(cfg)
        self.vib = VIB(self.encoder.feat_dim, cfg.latent_dim)
        self.decoder = _Decoder(cfg, self.encoder.feat_shape)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z, _mu, _logvar, kl_per_dim = self.vib(h)
        x_hat = self.decoder(z)
        return x_hat, kl_per_dim


def reconstruction_distortion(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Distortion D: squared error summed over pixels, averaged over the batch.

    Summing over pixels (not averaging) corresponds to the negative log
    likelihood of a fixed-variance Gaussian decoder, which keeps beta on the
    standard IB / beta-VAE scale relative to the per-sample KL.
    """
    se = (x - x_hat).pow(2).flatten(1).sum(dim=1)
    return se.mean()


def psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB for signals in [0, 1] (higher = better)."""
    mse = (x - x_hat).pow(2).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))
