"""Shared arbgrad-backed (spiral/rosette) building blocks for the zoo plugin
family.

Used by ``gre_noncart_2d.py`` and ``gre_noncart_3d.py``. Both trajectory
shapes share the same "single base waveform + per-shot rotation" machinery
(see ``pulserver.arbgrad`` module docstring): the C++ layer designs one
slew/gradient-limited shot in standard orientation; this module converts it
to SI units once and lets callers replay it per shot with a
``pulserver.pulseq.make_rotation`` extension, exactly like
``gre_radial_2d.py``/``gre_radial_3d.py`` do for plain trapezoid spokes.

Loaded via file-path (``importlib.util.spec_from_file_location``), same
convention as ``_gre_common.py``/``_prep_common.py``.
"""

from __future__ import annotations

import numpy as np
import pypulseq as pp

from pulserver import arbgrad

TRAJECTORY_SPIRAL = "spiral"
TRAJECTORY_ROSETTE = "rosette"


def trajectory_name(code: float) -> str:
    return TRAJECTORY_ROSETTE if code >= 0.5 else TRAJECTORY_SPIRAL


def order_mode_name(code: float) -> str:
    return "golden" if code >= 0.5 else "uniform"


def build_base_waveform(opts: pp.Opts, trajectory: str, fov_m: float, n_pix: int):
    """Design the single base (x, y) waveform in SI units (T/m).

    Returns ``(k0, grad_si, n_shots)`` where ``grad_si`` is ``(n_samp, 2)``
    (the z column is always zero for these in-plane trajectories — verified
    against the vendored MRArbGrad core — so it is dropped here; 3D callers
    add their own Cartesian partition-encode Z gradient separately).
    """
    slew_limit_hz_pix_s = opts.max_slew * fov_m / n_pix
    grad_limit_hz_pix = opts.max_grad * fov_m / n_pix
    dt_s = opts.grad_raster_time

    if trajectory == TRAJECTORY_ROSETTE:
        wf = arbgrad.rosette(fov_m, n_pix, slew_limit_hz_pix_s, grad_limit_hz_pix, dt_s)
    elif trajectory == TRAJECTORY_SPIRAL:
        wf = arbgrad.spiral(fov_m, n_pix, slew_limit_hz_pix_s, grad_limit_hz_pix, dt_s)
    else:
        raise ValueError(f"Unknown trajectory '{trajectory}' (expected 'spiral' or 'rosette')")

    grad_si = arbgrad.to_gradient_tesla_per_meter(wf, fov_m, n_pix, gamma=opts.gamma)
    return wf.k0, grad_si[:, :2], wf.n_shots


def build_readout_gradients(opts: pp.Opts, grad_si_xy: np.ndarray):
    """Build the (gx, gy) arbitrary-gradient events for the base waveform."""
    gx = pp.make_arbitrary_grad(channel="x", waveform=np.ascontiguousarray(grad_si_xy[:, 0]), system=opts)
    gy = pp.make_arbitrary_grad(channel="y", waveform=np.ascontiguousarray(grad_si_xy[:, 1]), system=opts)
    return gx, gy


def build_readout_adc(opts: pp.Opts, n_samples: int):
    """ADC spanning the whole base waveform; k-space center is the first
    sample (the waveform starts at ``k0`` ≈ 0), so there is no ADC delay."""
    return pp.make_adc(num_samples=n_samples, dwell=opts.grad_raster_time, delay=0.0, system=opts)
