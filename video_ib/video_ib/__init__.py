"""video_ib: a self-contained study of the Information Bottleneck on video.

Pipeline: encode a short video clip into a stochastic latent z, push z through
a Variational Information Bottleneck (VIB), and decode z back into the clip.
Varying the bottleneck strength beta lets us watch what information the latent
keeps versus discards -- i.e. the effect of the IB.

This package is intentionally independent of everything else in the repo.
"""

from .model import VIB, VideoIBAutoencoder
from .data import make_moving_shapes, MovingShapes

__all__ = ["VIB", "VideoIBAutoencoder", "make_moving_shapes", "MovingShapes"]
