"""Reconstruction for :mod:`pulserver.seqzoo.bssfp_2d`.

A balanced train samples the same Cartesian grid its spoiled counterpart
samples, with the same counters and boundaries -- the alternating RF phase is
already matched by the alternating ADC phase, so the data arrive demodulated
and the pipeline reads what was encoded without asking what steady state
produced it. So this is :class:`pulserver.reczoo.gre_2d.Gre2DRecon` under
the name the runtime pairs with the sequence.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Bssfp2DRecon"]

from pulserver.reczoo.gre_2d import Gre2DRecon


class Bssfp2DRecon(Gre2DRecon):
    """Reconstruct a 2D balanced SSFP scan, one image per slice."""


PLUGIN = Bssfp2DRecon()
