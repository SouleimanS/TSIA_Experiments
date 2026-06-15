"""Attention maps with vs without the information bottleneck.

Replicates the spirit of Figure 6 in Nagrani et al., "Attention Bottlenecks
for Multimodal Fusion" (NeurIPS 2021): compare the LLM's attention over visual
tokens for the vanilla (no-bottleneck) path and the VIB-bottleneck path, from
the same trained checkpoint, with quantitative localization metrics.

See plot_ib_attention.py.
"""
