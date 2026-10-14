"""Coil maps by nonlinear inversion (``bart nlinv -t``, with ``--phase-pole 0`` for a plane), then a wavelet-regularised non-Cartesian ``pics`` solve (``bart pics -t``), one image per slice and loop position."""

from pulserver.recon.handlers.nufft import PLUGIN  # noqa: F401
