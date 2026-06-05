"""Variant 6: sink-aware C-MIB with residual identity init.

Architecture variants (all share: no joint VIB, SinkAwareVIB on video):

    b_std_fusion       SinkAwareVIB_v + standard VIB_a + MutualCrossAttn
    b_topk_fusion      SinkAwareVIB_v + NormTopKSinkVIB_a + MutualCrossAttn
    b_std_nofusion     SinkAwareVIB_v + standard VIB_a + direct concat
    b_topk_nofusion    SinkAwareVIB_v + NormTopKSinkVIB_a + direct concat
    b_topk_fusion_adavib   b_topk_fusion + per-sample adaptive beta
    b_topk_fusion_adavib2  same arch, different adaptive_beta_base hyperparam
    a                  SinkAwareVIB_v + standard VIB_a + Fusion + SinkAwareVIB_joint
    c                  SinkAwareVIB_v + standard VIB_a + Fusion + standard VIB_joint

SinkAwareVIB design:
    - Classifies each token as sink or non-sink via phi = RMSNorm(x)[{985,1992}].abs().max >= 18
    - Two zero-init residual paths (mu = x at step 0, true identity):
        sink:     mu = x + W_sink·x,  logvar = 2·log_sigma_sink  (scalar, init=-10)
        non-sink: mu = x + W_ns·x,    logvar = b + W_lv·x        (b init=-3)
    - KL is split: kl = kl_nonsink + beta_sink_ratio * kl_sink
      Sinks get 100x less KL pressure — preserved, not compressed.

Six-term forward_train return:
    nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j
Trainer composes:
    loss = nll + beta_v*kl_v + beta_a*kl_a + beta_j*kl_j + aux_weight*(nll_aux_v + nll_aux_a)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from av_ib.backbone.qwen_omni import QwenOmniWrapper
from av_ib.model.bottleneck import VIB
from av_ib.model.fusion import MutualCrossAttention, SinkSymmetricFusion


# ---------------------------------------------------------------------------
# Sink classification constants (validated in Phase A)
# ---------------------------------------------------------------------------
DSINK_DIMS = [985, 1992]
SINK_TAU = 18.0
BETA_SINK_RATIO = 0.01   # sink KL weight relative to non-sink

ANSWER_VOCAB_LIST = [
    "yes", "no", "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten",
    "left", "right", "middle", "indoor", "outdoor", "simultaneously",
    "violin", "cello", "piano", "flute", "guitar", "clarinet", "saxophone", "accordion",
    "trumpet", "tuba", "trombone", "horn", "ukulele", "banjo", "pipa", "guzheng",
    "erhu", "suona", "xylophone", "drum", "congas", "bassoon", "bagpipe",
]


def _rmsnorm(x: Tensor, eps: float = 1e-6) -> Tensor:
    x32 = x.float()
    return x32 / (x32.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def _classify_sinks(x: Tensor, dsink_dims, tau: float) -> Tensor:
    """(B, T, D) -> (B, T) bool sink mask."""
    normed = _rmsnorm(x)
    phi = normed[..., dsink_dims].abs().max(dim=-1).values
    return phi >= tau


# ---------------------------------------------------------------------------
# SinkAwareVIB
# ---------------------------------------------------------------------------
class SinkAwareVIB(nn.Module):
    """Two-path residual VIB with sink/non-sink routing.

    At init both paths are identity (W=0) so step-0 output == input exactly.
    Sink tokens accumulate 1/beta_sink_ratio less KL pressure than non-sinks.

    Returns: z (B,T,D), kl_combined (scalar), sink_mask (B,T) bool
    """

    def __init__(self, d_model: int, dsink_dims=None, tau: float = SINK_TAU,
                 beta_sink_ratio: float = BETA_SINK_RATIO):
        super().__init__()
        self.d_model = d_model
        self.dsink_dims = list(dsink_dims) if dsink_dims is not None else list(DSINK_DIMS)
        self.tau = tau
        self.beta_sink_ratio = beta_sink_ratio

        # Non-sink path
        self.fc_mu_nonsink = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc_mu_nonsink.weight)
        nn.init.zeros_(self.fc_mu_nonsink.bias)
        self.fc_logvar_nonsink = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc_logvar_nonsink.weight)
        nn.init.constant_(self.fc_logvar_nonsink.bias, -3.0)

        # Sink path (scalar logvar so variance is uniform across sink dims)
        self.fc_mu_sink = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc_mu_sink.weight)
        nn.init.zeros_(self.fc_mu_sink.bias)
        self.log_sigma_sink = nn.Parameter(torch.tensor(-10.0))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        sink_mask = _classify_sinks(x, self.dsink_dims, self.tau)  # (B, T)

        mu_nonsink = x + self.fc_mu_nonsink(x)
        logvar_nonsink = self.fc_logvar_nonsink(x).clamp(min=-10.0, max=10.0)

        mu_sink = x + self.fc_mu_sink(x)
        logvar_sink = (2.0 * self.log_sigma_sink).expand_as(mu_sink).clamp(min=-10.0, max=10.0)

        mask_3d = sink_mask.unsqueeze(-1)
        mu = torch.where(mask_3d, mu_sink, mu_nonsink)
        logvar = torch.where(mask_3d, logvar_sink, logvar_nonsink)

        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu

        kl_per_token = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).mean(dim=-1)  # (B,T)
        kl_sink    = (kl_per_token *  sink_mask.float()).sum()
        kl_nonsink = (kl_per_token * (~sink_mask).float()).sum()
        kl_combined = kl_nonsink + self.beta_sink_ratio * kl_sink

        with torch.no_grad():
            self.last_stats = {
                "sink_frac":   sink_mask.float().mean().detach().float(),
                "kl_sink":     kl_sink.detach().float(),
                "kl_nonsink":  kl_nonsink.detach().float(),
                "std_nonsink": torch.exp(0.5 * logvar_nonsink).mean().detach().float(),
            }

        return z, kl_combined, sink_mask


# ---------------------------------------------------------------------------
# NormTopKSinkVIB  (audio variant — dimension-based sink invalid for audio)
# ---------------------------------------------------------------------------
class NormTopKSinkVIB(nn.Module):
    """Two-path VIB where sinks are the top-k% highest L2-norm tokens.

    Motivated by audio rho=-0.06 correlation with video sink dims {985,1992}:
    dimension-based classification has no predictive validity for audio.
    L2-norm top-k is a model-agnostic proxy.

    Returns: z (B,T,D), kl_combined (scalar), sink_mask (B,T) bool
    """

    def __init__(self, d_model: int, top_k_frac: float = 0.40,
                 beta_sink_ratio: float = BETA_SINK_RATIO):
        super().__init__()
        self.top_k_frac = top_k_frac
        self.beta_sink_ratio = beta_sink_ratio

        self.fc_mu_nonsink = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc_mu_nonsink.weight)
        nn.init.zeros_(self.fc_mu_nonsink.bias)
        self.fc_logvar_nonsink = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc_logvar_nonsink.weight)
        nn.init.constant_(self.fc_logvar_nonsink.bias, -3.0)

        self.fc_mu_sink = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc_mu_sink.weight)
        nn.init.zeros_(self.fc_mu_sink.bias)
        self.log_sigma_sink = nn.Parameter(torch.tensor(-10.0))

    def _topk_mask(self, x: Tensor) -> Tensor:
        norms = x.norm(dim=-1)  # (B, T)
        k = max(1, int(norms.shape[1] * self.top_k_frac))
        threshold = norms.kthvalue(norms.shape[1] - k + 1, dim=1, keepdim=True).values
        return norms >= threshold  # (B, T)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        sink_mask = self._topk_mask(x)

        mu_nonsink   = x + self.fc_mu_nonsink(x)
        logvar_nonsink = self.fc_logvar_nonsink(x).clamp(-10.0, 10.0)
        mu_sink      = x + self.fc_mu_sink(x)
        logvar_sink  = (2.0 * self.log_sigma_sink).expand_as(mu_sink).clamp(-10.0, 10.0)

        mask_3d = sink_mask.unsqueeze(-1)
        mu     = torch.where(mask_3d, mu_sink,     mu_nonsink)
        logvar = torch.where(mask_3d, logvar_sink, logvar_nonsink)

        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu

        kl_per_token = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).mean(dim=-1)
        kl_sink    = (kl_per_token *  sink_mask.float()).sum()
        kl_nonsink = (kl_per_token * (~sink_mask).float()).sum()

        with torch.no_grad():
            self.last_stats = {
                "sink_frac":   sink_mask.float().mean().detach().float(),
                "kl_sink":     kl_sink.detach().float(),
                "kl_nonsink":  kl_nonsink.detach().float(),
                "std_nonsink": torch.exp(0.5 * logvar_nonsink).mean().detach().float(),
            }

        return z, kl_nonsink + self.beta_sink_ratio * kl_sink, sink_mask


# ---------------------------------------------------------------------------
# AVModelV6
# ---------------------------------------------------------------------------

_B_VARIANTS = ("b_std_fusion", "b_topk_fusion", "b_std_nofusion", "b_topk_nofusion",
               "b_topk_fusion_adavib", "b_topk_fusion_adavib2")
_ALL_VARIANTS = ("a", "c") + _B_VARIANTS


class AVModelV6(nn.Module):
    """Sink-aware C-MIB.  See module docstring for variant descriptions."""

    D_MODEL: int = 2048

    def __init__(
        self,
        qwen_model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        kl_reduction: str = "mean_per_dim",
        fusion_heads: int = 8,
        fusion_ffn_mult: int = 1,
        fusion_n_blocks: int = 1,
        sink_dims: List[int] = None,
        sink_tau: float = SINK_TAU,
        beta_sink_ratio: float = BETA_SINK_RATIO,
        variant: str = "b_std_fusion",
        adaptive_beta: bool = False,
        adaptive_beta_base: float = 0.1,
    ):
        super().__init__()

        if variant not in _ALL_VARIANTS:
            raise ValueError(f"variant must be one of {_ALL_VARIANTS}, got {variant!r}")

        # AdaVIB variants force adaptive beta on
        if variant in ("b_topk_fusion_adavib", "b_topk_fusion_adavib2"):
            adaptive_beta = True

        self.variant = variant
        self.adaptive_beta = adaptive_beta
        self.adaptive_beta_base = adaptive_beta_base

        # Derived fusion/audio-vib flags from variant name
        self._use_topk_audio  = "topk" in variant
        self._use_fusion      = "nofusion" not in variant and variant not in ("a", "c")
        # variants a and c always use fusion (MCA)
        if variant in ("a", "c"):
            self._use_fusion = True

        # ── QWEN BACKBONE ────────────────────────────────────────────
        self.qwen = QwenOmniWrapper(
            model_path=qwen_model_path,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # ── VIDEO VIB (always SinkAwareVIB in v6) ───────────────────
        self.bottleneck_v = SinkAwareVIB(
            d_model=self.D_MODEL,
            dsink_dims=sink_dims,
            tau=sink_tau,
            beta_sink_ratio=beta_sink_ratio,
        )

        # ── AUDIO VIB ────────────────────────────────────────────────
        if self._use_topk_audio:
            self.bottleneck_a = NormTopKSinkVIB(
                d_model=self.D_MODEL,
                top_k_frac=0.40,
                beta_sink_ratio=beta_sink_ratio,
            )
        else:
            self.bottleneck_a = VIB(
                d_model=self.D_MODEL,
                kl_reduction=kl_reduction,
            )

        # ── FUSION ───────────────────────────────────────────────────
        if self._use_fusion:
            self.fusion = MutualCrossAttention(
                d_model=self.D_MODEL,
                n_heads=fusion_heads,
                ffn_mult=fusion_ffn_mult,
                n_blocks=fusion_n_blocks,
            )
        else:
            self.fusion = None

        # ── JOINT VIB (only for variants a and c) ───────────────────
        if variant == "a":
            self.bottleneck_joint = SinkAwareVIB(
                d_model=self.D_MODEL,
                dsink_dims=sink_dims,
                tau=sink_tau,
                beta_sink_ratio=beta_sink_ratio,
            )
        elif variant == "c":
            self.bottleneck_joint = VIB(
                d_model=self.D_MODEL,
                kl_reduction=kl_reduction,
            )
        else:  # all b variants
            self.bottleneck_joint = None

        # ── AUX HEADS ────────────────────────────────────────────────
        self.vocab_size = self.qwen.tokenizer.vocab_size
        self.aux_head_v = nn.Linear(self.D_MODEL, self.vocab_size, bias=False)
        self.aux_head_a = nn.Linear(self.D_MODEL, self.vocab_size, bias=False)

        # AdaVIB: cache first-token ids for the answer vocabulary
        if self.adaptive_beta:
            tok = self.qwen.tokenizer
            ids = []
            for w in ANSWER_VOCAB_LIST:
                t = tok(w, add_special_tokens=False, return_tensors="pt").input_ids
                if t.numel():
                    ids.append(int(t[0, 0].item()))
            self.register_buffer("_ans_vocab_ids",
                                 torch.tensor(sorted(set(ids)), dtype=torch.long))
        else:
            self._ans_vocab_ids = None

    # ----------------------------------------------------------------
    # AdaVIB helpers
    # ----------------------------------------------------------------
    def _adaptive_beta_from(self, z_pool: Tensor) -> Tensor:
        """Per-sample beta from entropy of z_pool projected onto answer vocab.
        z_pool: (B, D). Returns (B,) with floor at 1e-3."""
        embed = self.qwen.thinker_text_model.get_input_embeddings().weight  # (V, D)
        E_ans = embed[self._ans_vocab_ids.to(embed.device)]                 # (V_ans, D)
        z_pool = z_pool.to(embed.device).to(embed.dtype)
        logits = z_pool @ E_ans.T                                           # (B, V_ans)
        probs  = torch.softmax(logits, dim=-1)
        H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)                  # (B,)
        H_max = float(torch.log(torch.tensor(float(len(self._ans_vocab_ids)))))
        H_norm = (H / H_max).clamp(min=1e-4, max=0.999)
        return torch.clamp(-self.adaptive_beta_base * torch.log(H_norm), min=1e-3)

    # ----------------------------------------------------------------
    # Aux loss target: first answer token id per sample
    # ----------------------------------------------------------------
    def _first_answer_token_ids(self, answers: List[str], device) -> Tensor:
        ids = []
        for ans in answers:
            t = self.qwen.tokenizer(ans, add_special_tokens=False,
                                    return_tensors="pt").input_ids
            ids.append(int(t[0, 0].item()) if t.numel()
                       else self.qwen.tokenizer.pad_token_id)
        return torch.tensor(ids, device=device, dtype=torch.long)

    # ----------------------------------------------------------------
    # Provider factory
    # ----------------------------------------------------------------
    def _make_provider(self):
        kls, zs, masks = {}, {}, {}

        def provider(audio_out: Tensor, video_out: Tensor) -> Tensor:
            # Device migration only — keep the added C-MIB modules in fp32 and
            # cast the bf16 encoder outputs up to fp32 instead. Training these
            # small modules in bf16 with AdamW at lr=1e-4 silently drops updates
            # once weights grow past the bf16 ULP (~0.004*|w|); fp32 master
            # weights avoid that. The splicer casts z_j back to the LLM dtype
            # before masked_scatter, so the bf16 backbone is unaffected.
            target_device = video_out.device
            sample_param  = next(self.bottleneck_v.parameters())
            if sample_param.device != target_device:
                mods = [self.bottleneck_v, self.bottleneck_a, self.aux_head_v, self.aux_head_a]
                if self.fusion is not None:
                    mods.append(self.fusion)
                if self.bottleneck_joint is not None:
                    mods.append(self.bottleneck_joint)
                for mod in mods:
                    mod.to(device=target_device)  # device only; stay fp32

            # Cast bf16 encoder outputs up to fp32 so they match the fp32 modules
            video_out = video_out.float()
            audio_out = audio_out.float()

            # Video VIB (always SinkAwareVIB)
            z_v, kl_v, mask_v = self.bottleneck_v(video_out)
            masks["sink_v"] = mask_v

            # Audio VIB
            out_a = self.bottleneck_a(audio_out)
            z_a, kl_a = out_a[0], out_a[1]

            # AdaVIB: scale kl by per-sample entropy-derived beta
            if self.adaptive_beta:
                kl_v = kl_v * self._adaptive_beta_from(z_v.mean(1)).mean()
                kl_a = kl_a * self._adaptive_beta_from(z_a.mean(1)).mean()

            # Fusion
            if self.fusion is not None:
                z_v, z_a = self.fusion(z_v, z_a)

            av = torch.cat([z_v, z_a], dim=1)

            # Joint VIB
            if self.variant == "a":
                z_j, kl_j, mask_j = self.bottleneck_joint(av)
                masks["sink_j"] = mask_j
            elif self.variant == "c":
                z_j, kl_j = self.bottleneck_joint(av)
                masks["sink_j"] = None
            else:  # all b variants
                z_j = av
                kl_j = torch.zeros((), device=av.device, dtype=av.dtype)
                masks["sink_j"] = None

            kls["v"], kls["a"], kls["j"] = kl_v, kl_a, kl_j
            zs["v"], zs["a"] = z_v, z_a
            return z_j

        return provider, kls, zs, masks

    # ----------------------------------------------------------------
    # Training forward
    # ----------------------------------------------------------------
    def forward_train(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        answers: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Returns (nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j)."""
        provider, kls, zs, _ = self._make_provider()

        nll = self.qwen.forward_train(
            videos, audios, prompts, answers,
            av_token_provider=provider,
        )

        z_v, z_a = zs["v"], zs["a"]
        target = self._first_answer_token_ids(answers, device=nll.device)

        logits_v = self.aux_head_v(z_v.mean(dim=1))
        logits_a = self.aux_head_a(z_a.mean(dim=1))
        nll_aux_v = F.cross_entropy(logits_v, target.to(logits_v.device))
        nll_aux_a = F.cross_entropy(logits_a, target.to(logits_a.device))

        self.last_diagnostics = self._collect_diagnostics()

        return nll, nll_aux_v, nll_aux_a, kls["v"], kls["a"], kls["j"]

    def _collect_diagnostics(self) -> dict:
        """Flatten per-bottleneck last_stats into a logging-friendly dict.

        Surfaces the failure modes that beta=0 pilots would otherwise hide:
          sink_frac_{v,a} -> 0.0 means sink classification is inert (Issue #5)
          std_nonsink_{v,a}     -> reparam noise still injected at beta=0 (Issue #3/#7)
          kl_{sink,nonsink}_{v,a} -> raw KL split before beta_sink_ratio
        """
        out = {}
        for tag, mod in (("v", self.bottleneck_v), ("a", self.bottleneck_a)):
            stats = getattr(mod, "last_stats", None)
            if not stats:
                continue
            for k, val in stats.items():
                try:
                    out[f"{k}_{tag}"] = float(val.item())
                except (AttributeError, ValueError):
                    out[f"{k}_{tag}"] = float("nan")
        return out

    # ----------------------------------------------------------------
    # Generation forward
    # ----------------------------------------------------------------
    @torch.no_grad()
    def forward_generate(
        self,
        videos: List[str],
        audios: List[str],
        prompts: List[str],
        max_new_tokens: int = 20,
    ) -> List[str]:
        provider, _, _, _ = self._make_provider()
        return self.qwen.forward_generate(
            videos, audios, prompts,
            max_new_tokens=max_new_tokens,
            av_token_provider=provider,
        )
