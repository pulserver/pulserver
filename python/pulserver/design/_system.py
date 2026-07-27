"""System-limit and raster-timing helpers shared by all sequence builders."""

from __future__ import annotations

import copy
import math

import numpy as np
import pypulseq as pp

DEFAULT_BANDWIDTH_HZ_PX = 125_000.0
MAX_RASTER_SEARCH_STEPS = 50_000
MAX_GRAD_DERATE = 0.9
MAX_SLEW_DERATE = 0.9

def copy_event(event):
    """Copy a Pulseq event so its scalar fields can be re-pointed independently.

    Shallow on purpose. A module re-emits the same template thousands of times
    with a different ``phase_offset``, ``freq_offset`` or amplitude, and those
    are all plain floats -- the only large members are the sample arrays
    (``signal``, ``waveform``, ``tt``, ``t``), which every caller either leaves
    alone or *replaces* with a new array rather than writing into. Sharing them
    is therefore invisible, and it is what lets the shapes be registered once
    instead of recompressed per shot (see
    ``pulserver.pypulseq.Sequence._fast_register_rf``).

    The contract that makes this safe, for anything handing events to this
    function: never write into an event's arrays in place.
    """
    clone = type(event).__new__(type(event))
    clone.__dict__.update(event.__dict__)
    return clone


def scale_grad(gradient, scale: float):
    """``pypulseq.scale_grad`` without its pass-by-value copy cost.

    Same arithmetic and same fields as upstream (see
    :func:`pypulseq.scale_grad`); only the copy differs. Upstream's
    ``copy.copy`` goes through ``__reduce_ex__``, which costs an order of
    magnitude more than rebinding a namespace's ``__dict__`` -- and this runs
    once per phase-encode of every shot, which for a large 3D protocol is the
    single most-called function in the whole design pass.

    The ``system=`` limit checks upstream offers are not reproduced: no caller
    here passes them, because the templates being scaled were already solved
    against the system limits at the worst-case area.
    """
    scaled = copy_event(gradient)
    if gradient.type == "trap":
        scaled.amplitude = gradient.amplitude * scale
        scaled.flat_area = gradient.flat_area * scale
    else:
        scaled.waveform = gradient.waveform * scale
        scaled.first = gradient.first * scale
        scaled.last = gradient.last * scale
    scaled.area = gradient.area * scale
    # Upstream drops a stale cached library ID; ours never carry one, but a
    # template handed in from elsewhere might.
    if hasattr(scaled, "id"):
        del scaled.id
    return scaled


def apply_system_derates(
    opts: pp.Opts,
    *,
    grad_derate: float = MAX_GRAD_DERATE,
    slew_derate: float = MAX_SLEW_DERATE,
) -> None:
    """Derate the gradient and slew limits of ``opts`` in place.

    The undeviated base limits are cached on the ``opts`` object on first
    call, so calling this repeatedly is idempotent (the derate is always
    applied to the original base values, never compounded).

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits object, modified in place.
    grad_derate : float, optional
        Fraction of the base ``max_grad`` to allow.
    slew_derate : float, optional
        Fraction of the base ``max_slew`` to allow.

    Returns
    -------
    None
        ``opts`` is modified in place.

    Examples
    --------
    Derate a default system to 90 % gradient and slew:

    >>> import pypulseq as pp
    >>> from pulserver.design import _system as system
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> base = opts.max_grad
    >>> system.apply_system_derates(opts)
    >>> opts.max_grad == 0.9 * base
    True
    """
    if not hasattr(opts, "_pulserver_base_max_grad"):
        opts._pulserver_base_max_grad = float(opts.max_grad)
    if not hasattr(opts, "_pulserver_base_max_slew"):
        opts._pulserver_base_max_slew = float(opts.max_slew)

    opts.max_grad = opts._pulserver_base_max_grad * float(grad_derate)
    opts.max_slew = opts._pulserver_base_max_slew * float(slew_derate)


def quantize_readout_timing(
    nx_ro: int,
    target_bw_hz_px: float,
    grad_raster_s: float,
    adc_raster_s: float,
    min_flat_time_s: float,
) -> tuple[float, float]:
    """Find an ADC dwell and readout flat time compatible with both rasters.

    Searches for the smallest ADC-raster-multiple dwell whose total flat time
    (``nx_ro * dwell``) lands exactly on the gradient raster while staying as
    close as possible to the requested per-pixel bandwidth.

    Parameters
    ----------
    nx_ro : int
        Number of readout samples.
    target_bw_hz_px : float
        Requested receiver bandwidth per pixel (Hz).
    grad_raster_s : float
        Gradient raster time (s).
    adc_raster_s : float
        ADC raster time (s).
    min_flat_time_s : float
        Lower bound on the flat time (s), e.g. from the max-gradient limit.

    Returns
    -------
    dwell_s : float
        ADC dwell time (s), a multiple of ``adc_raster_s``.
    flat_time_s : float
        Readout flat time ``nx_ro * dwell_s`` (s), a multiple of
        ``grad_raster_s``.

    Examples
    --------
    Quantize a 256-sample readout at ~250 Hz/px bandwidth:

    >>> from pulserver.design import _system as system
    >>> dwell, flat = system.quantize_readout_timing(256, 62_500.0, 1e-5, 1e-7, 0.0)
    >>> abs(flat / 1e-5 - round(flat / 1e-5)) < 1e-9
    True
    """
    target_dwell = 1.0 / target_bw_hz_px
    m_target = max(1, round(target_dwell / adc_raster_s))

    m_min = max(m_target, int(np.ceil(min_flat_time_s / (nx_ro * adc_raster_s))))

    best_m = None
    best_cost = None

    for step in range(MAX_RASTER_SEARCH_STEPS):
        m = m_min + step
        flat = nx_ro * m * adc_raster_s
        ratio = flat / grad_raster_s
        if abs(ratio - round(ratio)) > 1e-9:
            continue
        cost = abs((m * adc_raster_s) - target_dwell)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_m = m
            if cost == 0.0:
                break

    if best_m is None:
        best_m = m_min

    dwell = best_m * adc_raster_s
    flat_time = nx_ro * dwell
    return dwell, flat_time


def round_to_raster(value_s: float, raster_s: float = 1e-5) -> float:
    """Round ``value_s`` to the nearest multiple of ``raster_s``.

    RF "center of mass" timings (e.g. an adiabatic hypsec pulse's weighted
    center) generally do not fall on a raster boundary, so delays computed
    from them must be re-aligned before use as block durations.

    Parameters
    ----------
    value_s : float
        Time to round (s).
    raster_s : float, optional
        Raster period (s).

    Returns
    -------
    float
        ``value_s`` rounded to the nearest raster multiple.

    Examples
    --------
    >>> from pulserver.design import _system as system
    >>> system.round_to_raster(1.23456e-3)
    0.00123
    """
    return round(value_s / raster_s) * raster_s


def ceil_to_raster(value_s: float, raster_s: float) -> float:
    """Round ``value_s`` up to the next multiple of ``raster_s``.

    A small tolerance keeps values already on the raster from being pushed
    up a full step by floating-point noise.

    Parameters
    ----------
    value_s : float
        Time to round (s).
    raster_s : float
        Raster period (s).

    Returns
    -------
    float
        Smallest raster multiple ``>= value_s`` (up to tolerance).

    Examples
    --------
    >>> from pulserver.design import _system as system
    >>> system.ceil_to_raster(1.01e-5, 1e-5)
    2e-05
    >>> system.ceil_to_raster(2e-5, 1e-5)
    2e-05
    """
    return math.ceil(value_s / raster_s - 1e-10) * raster_s
