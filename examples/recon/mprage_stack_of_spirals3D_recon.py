"""Reconstruction for :mod:`pulserver.app.sequence.mprage_stack_of_spirals3D_sequence`.

The stacked non-Cartesian pipeline of
:mod:`pulserver.app.recon.gre_stack_of_stars3D_recon`: inverse FFT along the
Cartesian partitions, then a model-based per-plane solve against the
trajectory each acquisition carries -- which is per-arm here, since the
arms are explicit. The inversion contrast is what the coherent centring
bakes in; nothing here needs to know the arms were written out rather than
rotated.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "MprageStackOfSpirals3DRecon"]

from pulserver.app.recon.gre_stack_of_stars3D_recon import GreStackOfStars3DRecon


class MprageStackOfSpirals3DRecon(GreStackOfStars3DRecon):
    """Reconstruct the explicit-arms spiral MPRAGE, one volume per measurement."""


PLUGIN = MprageStackOfSpirals3DRecon()
