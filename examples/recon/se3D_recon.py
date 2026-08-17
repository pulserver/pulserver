"""Reconstruction for :mod:`pulserver.app.sequence.se3D_sequence`.

A 3D spin echo samples the same ``(partition, line, readout)`` grid its
gradient-echo counterpart samples, with the same counters, calibration
rectangle and boundaries -- the pipeline reads what was encoded and never
asks what contrast produced it. So this is
:class:`pulserver.app.recon.gre3D_recon.Gre3DRecon` under the name the runtime pairs
with the sequence, and any divergence the spin echo ever needs starts by
overriding here rather than by copying the pipeline.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Se3DRecon"]

from pulserver.app.recon.gre3D_recon import Gre3DRecon


class Se3DRecon(Gre3DRecon):
    """Reconstruct a 3D Cartesian spin echo, one volume per measurement."""


PLUGIN = Se3DRecon()
