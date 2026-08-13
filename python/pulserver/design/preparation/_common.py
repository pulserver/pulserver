"""The two things a magnetization preparation almost always needs."""

from __future__ import annotations

__all__ = ["AXES", "half_passages", "spoiler_gradients"]

import numpy as np

from ... import pypulseq as pp

AXES = ("x", "y", "z")


def half_passages(
    system: pp.Opts,
    duration_s: float,
    *,
    adiabaticity: int = 8,
    dwell_s: float = 10e-6,
    pulse_type: str = "hypsec",
) -> tuple:
    """The adiabatic pair that tips magnetization down and puts it back.

    A half passage is one half of a full adiabatic sweep. Run from far
    off-resonance to on-resonance, it carries magnetization from ``+z`` into
    the transverse plane; run in reverse, it carries it back. Both are
    adiabatic, so above a threshold transmit amplitude neither cares what the
    amplitude actually is -- which is the whole reason a preparation uses them
    instead of a pair of hard 90s.

    The two are exact mirrors: the second is the first time-reversed and
    conjugated. That is what makes the phase the sweep accrues on the way down
    unwind on the way up, so what is stored is the magnetization's *magnitude*
    and not a phase that varied with transmit field.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    duration_s : float
        Duration of each half passage (s).
    adiabaticity : int, optional
        Sweep-rate margin over the adiabatic condition. The default is twice
        an inversion's, because a half passage has only half a sweep to
        converge in and is measurably worse at the inversion default.
    dwell_s : float, optional
        RF raster (s).
    pulse_type : str, optional
        Sweep family, as :func:`pypulseq.make_adiabatic_pulse` names them.

    Returns
    -------
    rf_prep : RfEvent
        The half passage that tips down.
    rf_store : RfEvent
        The reverse half passage that stores what is left back on ``z``.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.pypulseq as pp
    >>> from pulserver.design.preparation._common import half_passages
    >>> down, up = half_passages(pp.Opts(), 4e-3)
    >>> bool(np.allclose(np.asarray(down.signal), np.conj(np.asarray(up.signal))[::-1]))
    True
    """
    full = pp.make_adiabatic_pulse(
        pulse_type=pulse_type,
        duration=2.0 * duration_s,
        dwell=dwell_s,
        adiabaticity=adiabaticity,
        system=system,
        use="preparation",
    )
    sweep = np.asarray(full.signal)[: np.asarray(full.signal).size // 2]
    rf_prep = pp.make_arbitrary_rf(
        sweep,
        flip_angle=np.pi / 2.0,
        no_signal_scaling=True,
        dwell=dwell_s,
        system=system,
        use="preparation",
    )
    rf_store = pp.make_arbitrary_rf(
        np.conj(sweep[::-1]),
        flip_angle=np.pi / 2.0,
        no_signal_scaling=True,
        dwell=dwell_s,
        system=system,
        use="preparation",
    )
    return rf_prep, rf_store


def spoiler_gradients(
    system: pp.Opts, spoiling_cycles: float, voxel_size_m: float
) -> tuple:
    """A crusher on each axis, all winding the same dephasing.

    Three axes rather than one because what a preparation has to destroy is a
    whole excitation, not a residue: a single axis leaves everything with no
    variation along it untouched.

    Returns
    -------
    tuple
        ``(gx_spoil, gy_spoil, gz_spoil)``.
    """
    return tuple(
        pp.make_crusher(spoiling_cycles, voxel_size_m, axis, system=system)[0]
        for axis in AXES
    )
