"""Frozen video and audio encoders, reusing AVHBench's loading code.

We don't reimplement EVA-ViT or ImageBind here. AVHBench's
video_llama package already loads both correctly from the checkpoints
on disk, and we use it as a vendored dependency.

Module layout:
    VideoEncoder  -- wraps EVA-ViT-G. Input: (B, T, 3, H, W). Output: (B*T, 257, 1408).
    AudioEncoder  -- wraps ImageBind. Input: audio file paths or pre-loaded waveform.
                                       Output: (B, N_clips, 229, 1280).

Both are frozen on construction. Trainable params: 0 in either class.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Union, List

import torch
from torch import nn, Tensor


# Path to the AVHBench repo we reuse code from. Set once here so all
# imports resolve. If this path is wrong, the imports below will fail
# loudly at module load — easier to debug than silent failures later.
_AVHBENCH_ROOT = Path.home() / "SOULEIMAN_repo" / "datasets" / "AVHBench" / "AVHBench-Align-FT"
if str(_AVHBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_AVHBENCH_ROOT))

# Now we can import AVHBench's encoder builders.
from video_llama.models.eva_vit import create_eva_vit_g  # noqa: E402
from video_llama.models.ImageBind.models import imagebind_model  # noqa: E402
from video_llama.models.ImageBind.models.imagebind_model import ModalityType  # noqa: E402
from video_llama.models.ImageBind import data as imagebind_data  # noqa: E402


class VideoEncoder(nn.Module):
    """EVA-ViT-G video encoder, frozen.

    Input shape contract:
        x: (B, T, 3, H, W), float, normalized to ImageNet stats.
           T is the number of sampled frames per video (typically 8).
           H, W are 224 (the resolution EVA-ViT-G expects).

    Output shape:
        (B*T, 257, 1408)   -- 256 patch tokens + 1 CLS token per frame,
                              channel dim = EVA-ViT-G hidden size.
        The (B*T) flattening is convenient for the downstream Q-Former
        which doesn't care about the batch/time split.

    All parameters are frozen (requires_grad=False). The forward pass
    runs under torch.no_grad() to skip activation memory.
    """

    HIDDEN_SIZE: int = 1408
    NUM_TOKENS: int = 257     # 224/14=16, 16*16=256 patches + 1 cls
    IMG_SIZE: int = 224

    def __init__(
        self,
        ckpt_path: str = str(_AVHBENCH_ROOT / "models" / "eva_vit_g.pth"),
        precision: str = "fp16",
    ):
        super().__init__()
        # AVHBench's create_eva_vit_g hardcodes a *relative* path
        # ('./models/eva_vit_g.pth'), so it only finds the checkpoint when
        # the cwd is the AVHBench repo root. We chdir there for the call.
        import os
        _prev = os.getcwd()
        os.chdir(str(_AVHBENCH_ROOT))
        try:
            self.vit = create_eva_vit_g(
                img_size=self.IMG_SIZE,
                drop_path_rate=0.0,
                use_checkpoint=False,
                precision=precision,
            )
        finally:
            os.chdir(_prev)
        # AVHBench's loader expects the checkpoint at a specific path; we
        # already patched their code to use './models/eva_vit_g.pth' relative
        # to the AVHBench repo root, which is where the file actually is.
        # So `create_eva_vit_g` finds it automatically.

        # Freeze and switch to eval mode permanently.
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, 3, H, W) -> (B*T, 257, 1408)

        EVA-ViT has fp16 Conv/Linear weights but fp32 LayerNorm weights.
        Without autocast, LayerNorm output (fp32) hits fp16 Linear input
        and crashes. autocast(fp16) handles the cast for us; matches the
        wrapping AVHBench uses around their Q-Former calls.
        """
        if x.dim() != 5:
            raise ValueError(f"VideoEncoder expects (B,T,3,H,W), got {tuple(x.shape)}")
        b, t = x.shape[:2]
        x_flat = x.reshape(b * t, *x.shape[2:])  # (B*T, 3, H, W)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            feats = self.vit(x_flat)              # (B*T, 257, 1408)
        return feats


class AudioEncoder(nn.Module):
    """ImageBind audio encoder, frozen.

    Input shape contract:
        Two ways to call:

        (a) audio_paths: List[str] of length B. Each is a path to a video
            or audio file. We load and preprocess inside (same code path
            as AVHBench, which means the PyAV patch we added applies).
            Returns: (B, N_clips, 1024).

        (b) waveform: pre-loaded Tensor (B, N_clips, 1, mel_bins=128, time=204).
            Returns: (B, N_clips, 1024).

    For training we'll usually use (a). For unit tests we'll use (b).

    Frozen, eval mode, runs under no_grad.
    """

    HIDDEN_SIZE: int = 1024
    # ImageBind returns one pooled vector per audio clip. The token count
    # is therefore the number of clips (default 3 with the standard sampler).

    def __init__(
        self,
        ckpt_path: str = str(_AVHBENCH_ROOT / "models" / "imagebind_huge.pth"),
    ):
        super().__init__()
        # AVHBench's imagebind_huge() loads the whole multimodal model.
        # We only use the audio branch.
        # AVHBench's imagebind_huge returns (model, audio_feature_dim).
        self.imagebind, _audio_feat_dim = imagebind_model.imagebind_huge(pretrained=False)
        # Load the official ImageBind checkpoint.
        sd = torch.load(ckpt_path, map_location="cpu")
        self.imagebind.load_state_dict(sd, strict=True)

        for p in self.imagebind.parameters():
            p.requires_grad = False
        self.imagebind.eval()

    @torch.no_grad()
    def forward_from_paths(self, audio_paths: List[str], device: torch.device) -> Tensor:
        """Load audio files from disk, preprocess to mel-spec, encode.

        Returns: (B, N_clips, 1024)
        """
        # AVHBench's helper does the whole pipeline: PyAV decode -> resample
        # -> mel-spectrogram -> clip sampling -> normalize.
        audio_data = imagebind_data.load_and_transform_audio_data(
            audio_paths, device
        )  # (B, N_clips, 1, 128, 204)

        return self.forward_from_mel(audio_data)

    @torch.no_grad()
    def forward_from_mel(self, mel: Tensor) -> Tensor:
        """Encode pre-computed mel-spectrograms.

        mel shape: (B, N_clips, 1, mel_bins=128, time=204)
        Returns: (B, N_clips, 229, 1280)

        AVHBench calls get_audio_feature(...) which returns
        (audio_feature, modality_value). The first is the per-token
        feature pre-projection -- that's what their audio Q-Former consumes,
        so we return the same.
        """
        # mel is already 5D (B, N_clips, 1, 128, 204). get_audio_feature
        # detects this via `ndim >= 5` and handles the clip reshape internally.
        # Returns (audio_feature, modality_value); AVHBench uses the second
        # as the audio Q-Former's encoder_hidden_states.
        _, modality_value = self.imagebind.get_audio_feature(
            mel, modality_type=ModalityType.AUDIO
        )
        # modality_value: (B, N_clips, 1024).
        return modality_value


def freeze_count(module: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params) for sanity checks."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total
