"""Reconstruction for :mod:`pulserver.seqzoo.se_3d`.

A 3D spin echo samples the same ``(partition, line, readout)`` grid its
gradient-echo counterpart samples, with the same counters, calibration
rectangle and boundaries -- the pipeline reads what was encoded and never
asks what contrast produced it. So this is
:class:`pulserver.reczoo.gre_3d.Gre3DRecon` under the name the runtime pairs
with the sequence, and any divergence the spin echo ever needs starts by
overriding here rather than by copying the pipeline.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Se3DRecon"]

from pulserver.reczoo.gre_3d import Gre3DRecon


class Se3DRecon(Gre3DRecon):
    """Reconstruct a 3D Cartesian spin echo, one volume per measurement."""


PLUGIN = Se3DRecon()
