"""Pulseq Phasor design helpers."""

__all__ = [
    'make_blip',
    'make_crusher',
    'make_phasor',
]

from types import SimpleNamespace

import numpy as np

from .. import pulseq as pp


def make_phasor(
    channel: str,
    system: pp.Opts,
    fov_m: float,
    npix: int,
) -> SimpleNamespace:
    """
    Compute phase encoding gradient.

    Parameters
    ----------
    channel : str
        Orientation of phase encoding gradient event.
        Must be one of `x`, `y` or `z`.
    system : pp.Opts
        Pulseq system limits.
    fov_m : float
        Field of view along phase encoding direction in ``[m]``.
    npix : int
        Image size along phase encoding  direction.

    Returns
    -------
    SimpleNamespace
        Phase encoding gradient event.
    """
    delta_k = 1.0 / fov_m
    k_max = npix * delta_k

    # Design as extended trapezoid to achieve minimal duration
    phasor, _, _ = pp.make_extended_trapezoid_area(
        channel=channel,
        system=system,
        area=k_max,
        grad_start=0.0,
        grad_end=0.0,
    )

    # Convert to standard trapezoid
    rise_time, flat_time, fall_time = np.diff(phasor.tt)
    amplitude = max(phasor.waveform)

    return pp.make_trapezoid(
        channel=channel,
        system=system,
        amplitude=amplitude,
        rise_time=rise_time,
        flat_time=flat_time,
        fall_time=fall_time,
        delay=phasor.delay,
    )


def make_blip(
    channel: str,
    system: pp.Opts,
    fov_m: float,
    max_spacing: int = 1,
) -> SimpleNamespace:
    """
    Compute phase blip gradient.

    Parameters
    ----------
    channel : str
        Orientation of phase encoding gradient event.
        Must be one of `x`, `y` or `z`.
    system : pp.Opts
        Pulseq system limits.
    fov_m : float
        Field of view along phase encoding direction in ``[m]``.
    max_spacing : int, optional
        Maximum spacing between consectuvely acquired lines in units
        of ``1 / fov_m`` along the phase encoding direction

    Returns
    -------
    SimpleNamespace
        Phase blip gradient event.
    """
    delta_k = 1.0 / fov_m

    # Design as extended trapezoid to achieve minimal duration
    phasor, _, _ = pp.make_extended_trapezoid_area(
        channel=channel,
        system=system,
        area=max_spacing * delta_k,
        grad_start=0.0,
        grad_end=0.0,
    )

    # Convert to standard trapezoid
    rise_time, flat_time, fall_time = np.diff(phasor.tt)
    amplitude = max(phasor.waveform)

    return pp.make_trapezoid(
        channel=channel,
        system=system,
        amplitude=amplitude,
        rise_time=rise_time,
        flat_time=flat_time,
        fall_time=fall_time,
        delay=phasor.delay,
    )


def make_crusher(
    channel: str,
    system: pp.Opts,
    dephasing_angle_rad: float,
    thickness_m: float,
    start_area: float = 0.0,
    grad_first: float = 0.0,
    grad_last: float = 0.0,
) -> SimpleNamespace:
    """
    Compute general crusher.

    Parameters
    ----------
    channel : str
        Orientation of gradient event.
        Must be one of `x`, `y` or `z`.
    system : pp.Opts
        Pulseq system limits.
    dephasing_angle_rad : float
        Dephasing angle across given thickness in units of ``[rad]``.
    thickness_m : float
        Thickness of spoiled slab in ``[m]``.
    start_area : float, optional
        Initial k-space area in ``[Hz/m]``.
        The default is ``0.0``.
    grad_first : float, optional
        Initial gradient amplitude in ``[Hz/m/s]``.
        The default is ``0.0``.
    grad_last : float, optional
        Final gradient amplitude in ``[Hz/m/s]``.
        The default is ``0.0``.

    Returns
    -------
    SimpleNamespace
        Crusher gradient event.

    """
    # Compute target spoil area
    spoiling_area = pp.calc_spoil_area(dephasing_angle_rad, thickness_m)

    # Calculate target spoiler area
    spoiler_area = spoiling_area - start_area

    # Calculate gradient
    g_spoil, _, _ = pp.make_extended_trapezoid_area(
        channel=channel,
        system=system,
        area=spoiler_area,
        grad_start=grad_first,
        grad_end=grad_last,
    )

    return g_spoil
