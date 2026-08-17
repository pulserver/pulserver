"""Reconstruction for :mod:`pulserver.app.sequence.se_propeller2D_sequence`.

A blade is a rotated Cartesian strip, and a set of blades is a non-Cartesian
sampling of the disc -- so the model-based pipeline of
:mod:`pulserver.app.recon.gre_radial2D_recon` reconstructs it directly from the
trajectory each acquisition carries: density compensation over the
overlapping centres, adjoint-derived sensitivities, CG-SENSE. This example
does not implement PROPELLER's per-blade phase and motion correction; that
refinement starts by overriding here.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "SePropeller2DRecon"]

from pulserver.app.recon.gre_radial2D_recon import GreRadial2DRecon


class SePropeller2DRecon(GreRadial2DRecon):
    """Reconstruct a 2D PROPELLER spin echo, one image per slice."""


PLUGIN = SePropeller2DRecon()
