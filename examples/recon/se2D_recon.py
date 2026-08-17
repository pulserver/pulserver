"""Reconstruction for :mod:`pulserver.app.sequence.se2D_sequence`.

A spin echo samples the same Cartesian grid its gradient-echo counterpart
samples, with the same counters, calibration block and segment boundaries --
the pipeline reads what was encoded and never asks what contrast produced
it. So this is :class:`pulserver.app.recon.gre2D_recon.Gre2DRecon` under the name
the runtime pairs with the sequence, and any divergence the spin echo ever
needs starts by overriding here rather than by copying the pipeline.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Se2DRecon"]

from pulserver.app.recon.gre2D_recon import Gre2DRecon


class Se2DRecon(Gre2DRecon):
    """Reconstruct a 2D Cartesian spin echo, one image per slice."""


PLUGIN = Se2DRecon()
