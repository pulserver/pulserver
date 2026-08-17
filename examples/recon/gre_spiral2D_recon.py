"""Reconstruction for :mod:`pulserver.app.sequence.gre_spiral2D_sequence`.

The model-based non-Cartesian pipeline of
:mod:`pulserver.app.recon.gre_radial2D_recon`, which never asks what shape the
trajectory traces: density compensation, sensitivities and the CG-SENSE
operator are all built from the coordinates each acquisition carries. So
this is that plugin under the name the runtime pairs with the sequence.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "GreSpiral2DRecon"]

from pulserver.app.recon.gre_radial2D_recon import GreRadial2DRecon


class GreSpiral2DRecon(GreRadial2DRecon):
    """Reconstruct a 2D spiral gradient echo, one image per slice."""


PLUGIN = GreSpiral2DRecon()
