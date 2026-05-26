"""C-MIB splicer: forward hooks that inject VIB+fusion-processed tokens into
the Thinker's inputs_embeds before they reach the inner text LLM.

Strategy A2 (chosen after reading Qwen3OmniMoeThinkerForConditionalGeneration.forward
at transformers v5.x line 2041):

1. Capture (read-only):
     - get_audio_features  -> last_hidden_state          shape (N_a, 2048)
     - get_video_features  -> .pooler_output              shape (N_v, 2048)
     - get_placeholder_mask -> (image_mask, video_mask, audio_mask)
       fires twice (once after audio, once after video); we keep both

2. Splice (the one mutation point):
     - forward_pre_hook on self.model (inner text LLM)
     - reshape captured features (N, D) -> (1, N, D)  [batch-size-1 assumption]
     - call provider(audio_3d, video_3d) -> z_joint (1, N_a + N_v, 2048)
     - split z_joint back into z_a (1, N_a, D) and z_v (1, N_v, D)
     - flatten and write into inputs_embeds at audio_mask + video_mask positions
     - return modified args to self.model

The provider closure is read off `wrapper._current_provider`, which Phase 2
sets per forward_train/forward_generate call. If provider is None (e.g. caller
just wants vanilla Qwen behavior), splice does nothing.

Lifecycle:
    splicer = CMIBSplicer(wrapper)
    splicer.attach()
    # train / eval ...
    splicer.detach()   # in __del__ or test teardown
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor


class CMIBSplicer:
    """Hook-based splicer for Qwen3-Omni Thinker. See module docstring."""

    def __init__(self, wrapper):
        """`wrapper` is a QwenOmniWrapper. We reach into wrapper.model.thinker."""
        self.wrapper = wrapper
        thinker = wrapper.model.thinker
        self.thinker = thinker
        self.text_model = thinker.model  # inner LLM that consumes inputs_embeds

        # Per-call state (cleared at end of each splice or if no provider)
        self._audio_feats: Optional[Tensor] = None    # (N_a, D)
        self._video_feats: Optional[Tensor] = None    # (N_v, D)
        self._audio_mask: Optional[Tensor] = None     # mask into inputs_embeds
        self._video_mask: Optional[Tensor] = None     # mask into inputs_embeds
        self._mask_call_count: int = 0                # which call to get_placeholder_mask

        self._handles: list = []  # for detach()

    def attach(self) -> None:
        if self._handles:
            raise RuntimeError("Splicer already attached. Detach first.")
        self._handles.append(
            self.thinker.get_audio_features.__func__  # noqa: unused, placeholder
        )
        # We can't hook methods directly — they aren't nn.Modules. Hook the
        # encoder modules they wrap instead, or use a Python wrapper.
        # Cleanest path: wrap the methods themselves via descriptor swap.
        self._wrap_method("get_audio_features", self._on_audio_features)
        self._wrap_method("get_video_features", self._on_video_features)
        self._wrap_method("get_placeholder_mask", self._on_placeholder_mask)

        # forward_pre_hook on the inner text model — this IS an nn.Module
        h = self.text_model.register_forward_pre_hook(
            self._on_text_model_pre, with_kwargs=True,
        )
        self._handles.append(("text_model_pre_hook", h))

    def detach(self) -> None:
        # Restore wrapped methods
        for entry in list(self._handles):
            if isinstance(entry, tuple) and entry[0] == "method_swap":
                _, name, original = entry
                setattr(self.thinker, name, original)
            elif isinstance(entry, tuple) and entry[0] == "text_model_pre_hook":
                entry[1].remove()
        self._handles.clear()
        self._clear_per_call_state()

    # ------------------------------------------------------------------
    # Method-wrapping (since get_X_features are bound methods, not modules)
    # ------------------------------------------------------------------

    def _wrap_method(self, name: str, observer):
        """Replace self.thinker.<name> with a wrapper that calls observer(result)
        after the original method returns. Records swap for detach()."""
        original = getattr(self.thinker, name)

        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            observer(result)
            return result

        # Bind as bound method so signature is preserved
        setattr(self.thinker, name, wrapper)
        self._handles.append(("method_swap", name, original))

    # ------------------------------------------------------------------
    # Capture observers
    # ------------------------------------------------------------------

    def _on_audio_features(self, result) -> None:
        # get_audio_features returns BaseModelOutput-like; we want .last_hidden_state
        self._audio_feats = result.last_hidden_state

    def _on_video_features(self, result) -> None:
        # get_video_features returns BaseModelOutputWithDeepstackFeatures
        # We splice only on pooler_output (the main token stream); deepstack
        # multi-scale features flow untouched.
        self._video_feats = result.pooler_output

    def _on_placeholder_mask(self, result) -> None:
        # Returns (image_mask, video_mask, audio_mask)
        # Called twice in Thinker.forward — once after audio (we want audio_mask),
        # once after video (we want video_mask).
        image_mask, video_mask, audio_mask = result
        if self._mask_call_count == 0:
            self._audio_mask = audio_mask
        elif self._mask_call_count == 1:
            self._video_mask = video_mask
        # If called more than twice (shouldn't happen with AV input), ignore.
        self._mask_call_count += 1

    # ------------------------------------------------------------------
    # The actual splice
    # ------------------------------------------------------------------

    def _on_text_model_pre(self, module: nn.Module, args, kwargs):
        """Forward_pre_hook on self.thinker.model. Returns modified (args, kwargs).
        If provider is None or required captures are missing, returns unchanged."""

        provider = getattr(self.wrapper, "_current_provider", None)
        if provider is None:
            self._clear_per_call_state()
            return args, kwargs

        # inputs_embeds may be in args or kwargs depending on call site
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            inputs_embeds = kwargs["inputs_embeds"]
            inputs_embeds_key = ("kwargs", "inputs_embeds")
        else:
            # Hope it's the first positional arg if not in kwargs;
            # but Qwen always calls with kwargs so this branch is unlikely.
            self._clear_per_call_state()
            return args, kwargs

        # Need both modality captures present to do C-MIB splice
        if self._audio_feats is None or self._video_feats is None:
            # Not a full AV input (e.g. text-only or audio-only) — pass through
            self._clear_per_call_state()
            return args, kwargs

        if self._audio_mask is None or self._video_mask is None:
            # Mask capture failed (shouldn't happen with the standard forward)
            self._clear_per_call_state()
            return args, kwargs

        # Reshape flat captures (N, D) -> (1, N, D) for the VIB/fusion
        # which expect a batch dim. We're in batch-size-1 mode.
        audio_3d = self._audio_feats.unsqueeze(0)   # (1, N_a, D)
        video_3d = self._video_feats.unsqueeze(0)   # (1, N_v, D)
        n_a = audio_3d.size(1)
        n_v = video_3d.size(1)

        # Call provider — returns z_joint (1, N_v + N_a, D), order = [video, audio]
        # (matches v5._make_provider which does torch.cat([z_v_fused, z_a_fused], dim=1))
        z_joint = provider(audio_3d, video_3d)
        assert z_joint.size(1) == n_v + n_a, (
            f"Provider returned wrong token count: got {z_joint.size(1)}, "
            f"expected n_v + n_a = {n_v + n_a}"
        )

        # Split back
        z_v_new = z_joint[:, :n_v, :]      # (1, N_v, D)
        z_a_new = z_joint[:, n_v:, :]      # (1, N_a, D)
        # Flatten to (N, D) — the format masked_scatter wants
        z_v_flat = z_v_new.squeeze(0)
        z_a_flat = z_a_new.squeeze(0)

        # Match dtype/device of inputs_embeds
        z_v_flat = z_v_flat.to(inputs_embeds.device, inputs_embeds.dtype)
        z_a_flat = z_a_flat.to(inputs_embeds.device, inputs_embeds.dtype)

        # Write into inputs_embeds at the captured mask positions.
        # masked_scatter returns a new tensor; reassign.
        new_embeds = inputs_embeds.masked_scatter(self._audio_mask, z_a_flat)
        new_embeds = new_embeds.masked_scatter(self._video_mask, z_v_flat)

        new_kwargs = {**kwargs, "inputs_embeds": new_embeds}
        self._clear_per_call_state()
        return args, new_kwargs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_per_call_state(self) -> None:
        self._audio_feats = None
        self._video_feats = None
        self._audio_mask = None
        self._video_mask = None
        self._mask_call_count = 0

    def __del__(self):
        try:
            self.detach()
        except Exception:
            pass
