"""Coil maps by nonlinear inversion (``bart nlinv --phase-pole 0 -t``), then a wavelet-regularised non-Cartesian ``pics`` solve (``bart pics -t``), one image per slice and loop position."""

from pulserver.recon.handlers.nufft import PLUGIN  # noqa: F401
