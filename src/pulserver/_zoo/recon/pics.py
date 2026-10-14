"""Coil maps by nonlinear inversion (``bart nlinv``, with ``--phase-pole 0`` for a plane), then a wavelet-regularised ``pics`` solve (``bart pics``), one image per slice and loop position."""

from pulserver.recon.handlers.pics import PLUGIN  # noqa: F401
