"""Prompt-conditioned modality gating for audio-visual QA.

This subpackage adds a *modality gate* on top of the existing C-MIB pipeline.

Idea
----
Different questions need different modalities. "What colour is the singer's
shirt?" is answered from video; "Which instrument plays first?" is answered
from audio. A single fixed fusion of the two streams wastes capacity on the
irrelevant modality and lets it inject noise.

The gate reads the *prompt* (before any answer is generated), predicts a
probability distribution over modalities

    p = softmax(MLP(prompt_repr))            p = (p_video, p_audio, p_text)

and then *enhances* the injected modality tokens by scaling their embeddings
with the corresponding probability before they enter the LLM:

    z_video  <-  (M * p_video) * z_video
    z_audio  <-  (M * p_audio) * z_audio

The `M *` factor (M = number of modalities) centres the gate at the uniform
distribution: when p is uniform the scale is exactly 1.0, so an untrained gate
is the identity and the model degrades gracefully to the ungated baseline. The
gate then learns to up- or down-weight each modality *relative to uniform*.

The gate is trained end-to-end through the answer NLL — no extra loss term is
needed. Scaling the spliced embeddings changes what the LLM sees, which changes
the answer likelihood, which back-propagates into the gate MLP. The 30B
backbone stays frozen; only the gate (plus the existing LoRA + bottleneck
modules) receives gradients.

See model.py for AVModelGated and train.py for the training driver.
"""
from av_ib.modality_gate.gate import ModalityGate
from av_ib.modality_gate.model import AVModelGated

__all__ = ["ModalityGate", "AVModelGated"]
