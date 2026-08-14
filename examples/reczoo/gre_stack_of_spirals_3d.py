"""Reconstruction for :mod:`pulserver.seqzoo.gre_stack_of_spirals_3d`.

The stacked non-Cartesian pipeline of
:mod:`pulserver.reczoo.gre_stack_of_stars_3d`, which never asks what shape
the in-plane trajectory traces. So this is that plugin under the name the
runtime pairs with the sequence.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "GreStackOfSpirals3DRecon"]

from pulserver.reczoo.gre_stack_of_stars_3d import GreStackOfStars3DRecon


class GreStackOfSpirals3DRecon(GreStackOfStars3DRecon):
    """Reconstruct a stack of spirals, one volume per measurement."""


PLUGIN = GreStackOfSpirals3DRecon()
