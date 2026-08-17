"""Reconstruction for :mod:`pulserver.app.sequence.gre_stack_of_spirals3D_sequence`.

The stacked non-Cartesian pipeline of
:mod:`pulserver.app.recon.gre_stack_of_stars3D_recon`, which never asks what shape
the in-plane trajectory traces. So this is that plugin under the name the
runtime pairs with the sequence.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "GreStackOfSpirals3DRecon"]

from pulserver.app.recon.gre_stack_of_stars3D_recon import GreStackOfStars3DRecon


class GreStackOfSpirals3DRecon(GreStackOfStars3DRecon):
    """Reconstruct a stack of spirals, one volume per measurement."""


PLUGIN = GreStackOfSpirals3DRecon()
