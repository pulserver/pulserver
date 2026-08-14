"""Reconstruction for :mod:`pulserver.seqzoo.mprage_stack_of_spirals_3d`.

The stacked non-Cartesian pipeline of
:mod:`pulserver.reczoo.gre_stack_of_stars_3d`: inverse FFT along the
Cartesian partitions, then a model-based per-plane solve against the
trajectory each acquisition carries -- which is per-arm here, since the
arms are explicit. The inversion contrast is what the coherent centring
bakes in; nothing here needs to know the arms were written out rather than
rotated.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "MprageStackOfSpirals3DRecon"]

from pulserver.reczoo.gre_stack_of_stars_3d import GreStackOfStars3DRecon


class MprageStackOfSpirals3DRecon(GreStackOfStars3DRecon):
    """Reconstruct the explicit-arms spiral MPRAGE, one volume per measurement."""


PLUGIN = MprageStackOfSpirals3DRecon()
