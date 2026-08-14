"""Reconstruction for :mod:`pulserver.seqzoo.gre_spiral_2d`.

The model-based non-Cartesian pipeline of
:mod:`pulserver.reczoo.gre_radial_2d`, which never asks what shape the
trajectory traces: density compensation, sensitivities and the CG-SENSE
operator are all built from the coordinates each acquisition carries. So
this is that plugin under the name the runtime pairs with the sequence.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "GreSpiral2DRecon"]

from pulserver.reczoo.gre_radial_2d import GreRadial2DRecon


class GreSpiral2DRecon(GreRadial2DRecon):
    """Reconstruct a 2D spiral gradient echo, one image per slice."""


PLUGIN = GreSpiral2DRecon()
