"""Reconstruction for :mod:`pulserver.app.sequence.bssfp3D_sequence`.

A balanced train samples the same ``(partition, line, readout)`` grid its
spoiled counterpart samples, with the same counters and boundaries; the
alternating RF phase is already matched by the alternating ADC phase. So
this is :class:`pulserver.app.recon.gre3D_recon.Gre3DRecon` under the name the
runtime pairs with the sequence.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Bssfp3DRecon"]

from pulserver.app.recon.gre3D_recon import Gre3DRecon


class Bssfp3DRecon(Gre3DRecon):
    """Reconstruct a 3D balanced SSFP scan, one volume per measurement."""


PLUGIN = Bssfp3DRecon()
