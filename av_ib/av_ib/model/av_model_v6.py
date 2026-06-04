"""Variant 6 (Direction A): sink-aware C-MIB with residual identity init.

Key design difference from v5:
    - Sink classification on encoder output via Dsink={985, 1992}, tau=18
    - For each VIB (VIB_v and VIB_joint), TWO residual transforms:
        sink path:    mu = x + W_sink @ x       (W_sink init=0 -> identity)
                      logvar = b_sink (scalar)   (init=-10 -> tiny sigma)
        non-sink:     mu = x + W_nonsink @ x    (W_nonsink init=0 -> identity)
                      logvar = b_lv + W_lv @ x  (init=-3, standard VIB behavior)
    - VIB_joint re-classifies sinks on post-fusion features (NOT inherited)
    - VIB_a is standard (no sink awareness; audio rho was weak in Phase A)

At init: mu == x (true identity) and sink sigma ~ 0.007. So step-0 v6 behaves
exactly like vanilla Qwen3-Omni splicing the encoder output unchanged.
Training shapes the perturbations.

Six-term forward_train return: nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j
Trainer composes as in v5: loss = nll + beta_v*kl_v + beta_a*kl_a + beta_j*kl_j
                                  + aux_weight * (nll_aux_v + nll_aux_a)
v6 internally weights sink vs non-sink KL via beta_sink_ratio; trainer multiplies
the returned combined kl by beta_v/beta_j as usual.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from av_ib.backbone.qwen_omni import QwenOmniWrapper
from av_ib.model.bottleneck import VIB
from av_ib.model.fusion import MutualCrossAttention, SinkSymmetricFusion


# Sink classification constants (from Phase A validation)
DSINK_DIMS = [985, 1992]

ANSWER_VOCAB_LIST = [
    "yes","no","zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "left","right","middle","indoor","outdoor","simultaneously",
    "violin","cello","piano","flute","guitar","clarinet","saxophone","accordion",
    "trumpet","tuba","trombone","horn","ukulele","banjo","pipa","guzheng","erhu","suona","xylophone",
    "drum","congas","bassoon","bagpipe",
]

SINK_TAU = 18.0
BETA_SINK_RATIO = 0.01   # sink KL weight relative to non-sink


def _rmsnorm(x, eps=1e-6):
    """RMSNorm along last dim, in float32 for numerical stability."""
    x32 = x.float()
    return x32 / (x32.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def _classify_sinks(x: Tensor, dsink_dims, tau: float) -> Tensor:
    """Given (B, T, D) hidden state, return (B, T) bool sink mask."""
    normed = _rmsnorm(x)
    dsink_proj = normed[..., dsink_dims].abs()
    phi = dsink_proj.max(dim=-1).values
    return phi >= tau


class SinkAwareVIB(nn.Module):
    """Two-path residual VIB.

    For each token, classify sink vs non-sink based on phi, then route through
    one of two residual transforms. At init all transforms are identity (W=0).

    The KL contribution is split between sinks and non-sinks; sinks contribute
    a tiny fraction (beta_sink_ratio) of their KL because we want to PROTECT
    sinks, not compress them.
    """

    def __init__(self, d_model: int, dsink_dims=None, tau: float = SINK_TAU,
                 beta_sink_ratio: float = BETA_SINK_RATIO):
        super().__init__()
        self.d_model = d_model
        self.dsink_dims = list(dsink_dims) if dsink_dims is not None else list(DSINK_DIMS)
        self.tau = tau
        self.beta_sink_ratio = beta_sink_ratio

        # === Non-sink path ===
        # mu = x + W_nonsink @ x; W_nonsink init=0 so mu=x at step 0
        self.fc_mu_nonsink = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc_mu_nonsink.weight)
        nn.init.zeros_(self.fc_mu_nonsink.bias)
        # logvar = b + W_lv @ x; same init pattern, bias to -3 (matches v5)
        self.fc_logvar_nonsink = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc_logvar_nonsink.weight)
        nn.init.constant_(self.fc_logvar_nonsink.bias, -3.0)

        # === Sink path ===
        # mu = x + W_sink @ x; W init=0 so mu=x at step 0 (true preservation)
        self.fc_mu_sink = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.fc_mu_sink.weight)
        nn.init.zeros_(self.fc_mu_sink.bias)
        # logvar = single learnable scalar bias (NO per-input dependence) initialized very negative
        # so sigma starts at ~0.007 and grows only if training explicitly wants noise on sinks.
        self.log_sigma_sink = nn.Parameter(torch.tensor(-10.0))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x: (B, T, D)
        Returns: z (B, T, D), kl_combined (scalar, raw, unweighted by beta),
                 sink_mask (B, T) bool — useful for inspection
        """
        # Sink classification on input x (the VIB's input, not the encoder original)
        sink_mask = _classify_sinks(x, self.dsink_dims, self.tau)  # (B, T)

        # === Non-sink path ===
        mu_nonsink = x + self.fc_mu_nonsink(x)
        logvar_nonsink = self.fc_logvar_nonsink(x).clamp(min=-10.0, max=10.0)

        # === Sink path ===
        mu_sink = x + self.fc_mu_sink(x)
        # Broadcast scalar log_sigma_sink to per-element logvar (= 2 * log_sigma)
        logvar_sink = (2.0 * self.log_sigma_sink).expand_as(mu_sink).clamp(min=-10.0, max=10.0)

        # Route per-token via sink mask
        mask_3d = sink_mask.unsqueeze(-1)
        mu = torch.where(mask_3d, mu_sink, mu_nonsink)
        logvar = torch.where(mask_3d, logvar_sink, logvar_nonsink)

        # Reparameterization (only when training)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        # KL per-element, then per-token (mean over D), then split by sink/non-sink
        kl_per_elem = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)  # (B, T, D)
        kl_per_token = kl_per_elem.mean(dim=-1)  # (B, T)
        kl_sink = (kl_per_token * sink_mask.float()).sum()
        kl_nonsink = (kl_per_token * (~sink_mask).float()).sum()
        kl_combined = kl_nonsink + self.beta_sink_ratio * kl_sink

        return z, kl_combined, sink_mask




class NormTopKSinkVIB(nn.Module):
    """SinkAwareVIB variant that classifies sinks by L2 norm (top-k fraction).

    No dim dependency — the top-k highest-norm tokens are treated as sinks and
    get low KL pressure (beta_sink_ratio). Used for audio in v6b-bis where
    dimension-based sink classification has no predictive validity (rho=-0.06).
    """
    def __init__(self, d_model: int, top_k_frac: float = 0.40,
                 beta_sink_ratio: float = BETA_SINK_RATIO):
        super().__init__()
        self.d_model = d_model
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

    def _norm_topk_mask(self, x: Tensor) -> Tensor:
        """(B, T, D) -> (B, T) bool, True for top-k% highest L2-norm tokens."""
        norms = x.norm(dim=-1)          # (B, T)
        k = max(1, int(norms.shape[1] * self.top_k_frac))
        threshold = norms.kthvalue(norms.shape[1] - k + 1, dim=1, keepdim=True).values
        return norms >= threshold        # (B, T)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        sink_mask = self._norm_topk_mask(x)          # (B, T)
        mu_nonsink   = x + self.fc_mu_nonsink(x)
        logvar_nonsink = self.fc_logvar_nonsink(x).clamp(-10.0, 10.0)
        mu_sink      = x + self.fc_mu_sink(x)
        logvar_sink  = (2.0 * self.log_sigma_sink).expand_as(mu_sink).clamp(-10.0, 10.0)
        mask_3d = sink_mask.unsqueeze(-1)
        mu     = torch.where(mask_3d, mu_sink,     mu_nonsink)
        logvar = torch.where(mask_3d, logvar_sink, logvar_nonsink)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z   = mu + std * torch.randn_like(std)
        else:
            z = mu
        kl_per_token = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).mean(dim=-1)
        kl_sink    = (kl_per_token *  sink_mask.float()).sum()
        kl_nonsink = (kl_per_token * (~sink_mask).float()).sum()
        return z, kl_nonsink + self.beta_sink_ratio * kl_sink, sink_mask

class AVModelV6(nn.Module):
    """Sink-aware C-MIB: VIB_v and VIB_joint are SinkAwareVIB, VIB_a is standard VIB."""

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
        variant: str = "a",
        video_vib: str = "sink",
        audio_vib: str = "standard",
        adaptive_beta: bool = False,
        adaptive_beta_base: float = 0.1,
        fusion_type: str = "mutual",
    ):
        super().__init__()
        assert variant in ("a", "b", "b_bis", "b_bis_2", "b_bis_3", "b_bis_4", "c"), f"variant must be a/b/b_bis/b_bis_2/b_bis_3/b_bis_4/c, got {variant!r}"
        if variant in ("b_bis_3", "b_bis_4"):
            adaptive_beta = True
        assert video_vib in ("sink", "standard"), f"video_vib must be sink/standard, got {video_vib!r}"
        assert fusion_type in ("mutual", "sink_sym", "none"), f"fusion_type must be mutual/sink_sym/none, got {fusion_type!r}"
        self.variant = variant
        self.video_vib = video_vib
        self.audio_vib = audio_vib
        self.adaptive_beta = adaptive_beta
        self.adaptive_beta_base = adaptive_beta_base
        self.fusion_type = fusion_type

        self.qwen = QwenOmniWrapper(
            model_path=qwen_model_path,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        if video_vib == "sink":
            self.bottleneck_v = SinkAwareVIB(
                d_model=self.D_MODEL,
                dsink_dims=sink_dims,
                tau=sink_tau,
                beta_sink_ratio=beta_sink_ratio,
            )
        else:  # standard VIB on video (no sink-awareness) — isolation ablation
            self.bottleneck_v = VIB(d_model=self.D_MODEL, kl_reduction=kl_reduction)
        if audio_vib == "norm_topk":
            self.bottleneck_a = NormTopKSinkVIB(
                d_model=self.D_MODEL,
                top_k_frac=0.40,
                beta_sink_ratio=beta_sink_ratio,
            )
        else:  # standard VIB on audio
            self.bottleneck_a = VIB(d_model=self.D_MODEL, kl_reduction=kl_reduction)

        if fusion_type == "mutual":
            self.fusion = MutualCrossAttention(
                d_model=self.D_MODEL,
                n_heads=fusion_heads,
                ffn_mult=fusion_ffn_mult,
                n_blocks=fusion_n_blocks,
            )
        elif fusion_type == "sink_sym":  # sink-mediated symmetric fusion, zero-init residual
            self.fusion = SinkSymmetricFusion(
                d_model=self.D_MODEL,
                n_heads=fusion_heads,
                dsink_dims=sink_dims,
                sink_tau=sink_tau,
            )
        else:  # none: no fusion, direct concat
            self.fusion = None

        # Variant determines joint VIB:
        #   a: SinkAwareVIB (re-classifies sinks post-fusion)
        #   b: None (no joint bottleneck, fusion output goes directly to LLM)
        #   c: standard VIB (joint compression but no sink-awareness)
        if variant == "a":
            self.bottleneck_joint = SinkAwareVIB(
                d_model=self.D_MODEL,
                dsink_dims=sink_dims,
                tau=sink_tau,
                beta_sink_ratio=beta_sink_ratio,
            )
        elif variant in ("b", "b_bis", "b_bis_2", "b_bis_3", "b_bis_4"):
            self.bottleneck_joint = None
        else:  # variant == "c"
            self.bottleneck_joint = VIB(d_model=self.D_MODEL, kl_reduction=kl_reduction)

        self.vocab_size = self.qwen.tokenizer.vocab_size
        self.aux_head_v = nn.Linear(self.D_MODEL, self.vocab_size, bias=False)
        self.aux_head_a = nn.Linear(self.D_MODEL, self.vocab_size, bias=False)
        # AdaVIB: cache first-token ids for the answer vocab
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



    def _adaptive_beta_from(self, z_pool: Tensor) -> Tensor:
        """AdaVIB-style: compute per-sample beta from entropy of z_pool aligned to answer vocab.
        z_pool: (B, D)
        Returns: (B,) scalar betas with a floor to prevent collapse to 0.
        """
        # Get the LLM input embedding (D = hidden_size)
        embed = self.qwen.thinker_text_model.get_input_embeddings().weight  # (V, D)
        E_ans = embed[self._ans_vocab_ids.to(embed.device)]                 # (V_ans, D)
        z_pool = z_pool.to(embed.device).to(embed.dtype)
        logits = z_pool @ E_ans.T                                           # (B, V_ans)
        probs  = torch.softmax(logits, dim=-1)
        H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)                  # (B,)
        H_max = float(torch.log(torch.tensor(float(len(self._ans_vocab_ids)))))
        H_norm = (H / H_max).clamp(min=1e-4, max=0.999)
        beta = -self.adaptive_beta_base * torch.log(H_norm)                 # (B,)
        # Floor: never drop below beta_min so KL always contributes
        beta_min = 1e-3
        beta = torch.clamp(beta, min=beta_min)
        return beta

    def _make_provider(self):
        """Provider that records sink masks and KLs for inspection."""
        kls, zs, masks = {}, {}, {}

        def provider(audio_out: Tensor, video_out: Tensor) -> Tensor:
            # Lazy device/dtype migration for added modules (Qwen uses device_map="auto")
            target_device = video_out.device
            target_dtype = video_out.dtype
            sample_param = next(self.bottleneck_v.parameters())
            needs_move = (sample_param.device != target_device or
                          sample_param.dtype != target_dtype)
            if needs_move:
                mods = [self.bottleneck_v, self.bottleneck_a] + ([self.fusion] if self.fusion is not None else []) + [
                        self.aux_head_v, self.aux_head_a]
                if self.bottleneck_joint is not None:
                    mods.append(self.bottleneck_joint)
                for mod in mods:
                    mod.to(device=target_device, dtype=target_dtype)

            # === VIB_v: sink-aware OR standard depending on video_vib ===
            if self.video_vib == "sink":
                z_v, kl_v, mask_v = self.bottleneck_v(video_out)
                masks["sink_v"] = mask_v
            else:
                z_v, kl_v = self.bottleneck_v(video_out)
                masks["sink_v"] = None

            # === VIB_a: standard ===
            _a_out = self.bottleneck_a(audio_out); z_a, kl_a = _a_out[0], _a_out[1]
            # === AdaVIB: per-sample adaptive beta scaling ===
            if self.adaptive_beta:
                z_v_pool = z_v.mean(dim=1)                                   # (B, D)
                z_a_pool = z_a.mean(dim=1)
                beta_v_adapt = self._adaptive_beta_from(z_v_pool).mean()    # scalar
                beta_a_adapt = self._adaptive_beta_from(z_a_pool).mean()
                kl_v = kl_v * beta_v_adapt
                kl_a = kl_a * beta_a_adapt
                # Stash for logging
                kls["beta_v_adapt"] = beta_v_adapt.detach()
                kls["beta_a_adapt"] = beta_a_adapt.detach()

            # === Fusion ===
            if self.fusion is not None:
                z_v_fused, z_a_fused = self.fusion(z_v, z_a)
            else:
                z_v_fused, z_a_fused = z_v, z_a
            av = torch.cat([z_v_fused, z_a_fused], dim=1)

            # === VIB_joint: variant-dependent ===
            if self.variant == "a":
                z_joint, kl_j, mask_j = self.bottleneck_joint(av)
                masks["sink_j"] = mask_j
            elif self.variant in ("b", "b_bis", "b_bis_2", "b_bis_3", "b_bis_4"):
                z_joint = av                                       # no bottleneck
                kl_j = torch.zeros((), device=av.device, dtype=av.dtype)
                masks["sink_j"] = None
            else:  # "c"
                z_joint, kl_j = self.bottleneck_joint(av)          # standard VIB
                masks["sink_j"] = None

            kls["v"], kls["a"], kls["j"] = kl_v, kl_a, kl_j
            zs["v"], zs["a"] = z_v, z_a
            return z_joint

        return provider, kls, zs, masks

    def _first_answer_token_ids(self, answers: List[str], device) -> Tensor:
        ids = []
        for ans in answers:
            t = self.qwen.tokenizer(ans, add_special_tokens=False, return_tensors="pt").input_ids
            ids.append(int(t[0, 0].item()) if t.numel() else self.qwen.tokenizer.pad_token_id)
        return torch.tensor(ids, device=device, dtype=torch.long)

    def forward_train(
        self,
        videos,
        audios,
        prompts: List[str],
        answers: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Returns 6-tuple compatible with v5 trainer.
        kl_v and kl_j combine sink and non-sink contributions internally via
        beta_sink_ratio. Trainer multiplies returned KLs by beta_v/beta_a/beta_j.
        """
        provider, kls, zs, masks = self._make_provider()
        nll = self.qwen.forward_train(
            videos, audios, prompts, answers, av_token_provider=provider,
        )
        kl_v, kl_a, kl_j = kls["v"], kls["a"], kls["j"]
        z_v, z_a = zs["v"], zs["a"]
        target = self._first_answer_token_ids(answers, device=nll.device)
        logits_v = self.aux_head_v(z_v.mean(dim=1))
        logits_a = self.aux_head_a(z_a.mean(dim=1))
        nll_aux_v = F.cross_entropy(logits_v, target.to(logits_v.device))
        nll_aux_a = F.cross_entropy(logits_a, target.to(logits_a.device))
        return nll, nll_aux_v, nll_aux_a, kl_v, kl_a, kl_j

    @torch.no_grad()
    def forward_generate(
        self,
        videos,
        audios,
        prompts: List[str],
        max_new_tokens: int = 10,
    ) -> List[str]:
        provider, _, _, _ = self._make_provider()
        return self.qwen.forward_generate(
            videos, audios, prompts,
            max_new_tokens=max_new_tokens, av_token_provider=provider,
        )

    def trainable_summary(self) -> dict:
        out = {}
        for name, child in self.named_children():
            n = sum(p.numel() for p in child.parameters() if p.requires_grad)
            out[name] = n
        out["__total_trainable__"] = sum(out.values())
        out["__total_params__"] = sum(p.numel() for p in self.parameters())
        return out
