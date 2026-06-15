"""Two fixes for the audio attention suppression diagnosed by token_analysis:

  Option 1 — token-count equalization: scale audio tokens by sqrt(n_v/n_a)
             so audio's total attention budget equals video's.

  Option 2 — direction-preservation loss: penalise cosine distance between
             z_a and audio_out so the VIB doesn't rotate audio into a
             subspace that k_proj cannot hear.

Both are implemented in AVModelBalanced.
"""
