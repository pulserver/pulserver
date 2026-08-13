"""Excitation of one slice or one slab."""

from __future__ import annotations

__all__ = ["SpatialSelectiveExcitation"]

import numpy as np

from ... import pypulseq as pp
from ._base import RfModule, rf_reference

_AXES = ("x", "y", "z")


class SpatialSelectiveExcitation(RfModule):
    """An SLR pulse under a selection gradient, with its rephaser.

    SLR rather than a windowed sinc: it turns the slice profile into a filter
    design problem, so the passband and stopband ripple are asked for rather
    than discovered, and the profile is far squarer at the same
    time-bandwidth product.

    ``is_slab`` decides how the rephaser is delivered, and the choice matters
    to whatever plays this module:

    - ``False`` (a 2D slice) publishes ``gz`` and ``gz_reph`` and lays out
      **two** blocks. The rephaser is separable, so a readout can fold its
      moment into its own prewinder and the excitation can then be played
      without it.
    - ``True`` (a 3D slab) concatenates the two into ``gz`` alone and lays out
      **one** block. A slab excitation is played once per TR next to a
      partition encode that occupies the same axis anyway, so nothing is saved
      by keeping the rephaser separate, and one gradient is one event for the
      readout to accept.

    To skip the rephasing entirely — a spin echo whose refocusing pulse
    re-winds the selection lobe — pass ``rephase=False``.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    flip_angle_deg : float
        Nominal flip angle (degrees).
    thickness_m : float
        Slice or slab thickness (m).
    duration_s : float, optional
        Pulse duration (s).
    is_slab : bool, optional
        Merge the rephaser into the selection gradient, as above.
    rephase : bool, optional
        Build a rephaser at all.
    time_bw_product : float, optional
        Time-bandwidth product. Higher is a squarer profile and a longer pulse
        at the same bandwidth.
    axis : {'z', 'x', 'y'}, optional
        Selection axis.
    pulse_type : {'st', 'ex', 'se', 'inv', 'sat'}, optional
        SLR design family. ``"ex"`` for a large-tip excitation, ``"se"`` for a
        refocusing pulse.
    use : str, optional
        What the pulse is for; the trajectory core reads it.
    passband_ripple, stopband_ripple : float, optional
        Ripple allowed in each band of the slice profile.

    Attributes
    ----------
    rf : RfEvent
        The pulse. Its ``delay`` already places it on the gradient's flat top.
    gz : TrapEvent or GradEvent
        The selection gradient; under ``is_slab`` the rephaser is part of it.
    gz_reph : TrapEvent
        The rephaser. Only when not ``is_slab`` and ``rephase``.

    Raises
    ------
    ValueError
        If ``thickness_m`` or ``duration_s`` is not positive, or ``axis`` is
        not a gradient channel.

    Examples
    --------
    A 2D slice keeps its rephaser separate, in a second block:

    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> slice_ = design.SpatialSelectiveExcitation(pp.Opts(), 15.0, 5e-3)
    >>> len(slice_.blocks), slice_.gz_reph.channel
    (2, 'z')

    A slab merges it, so the whole excitation is one block and one gradient:

    >>> slab = design.SpatialSelectiveExcitation(pp.Opts(), 8.0, 0.12, is_slab=True)
    >>> len(slab.blocks), slab.gz.type
    (1, 'grad')
    """

    def init_module(
        self,
        system: pp.Opts,
        flip_angle_deg: float,
        thickness_m: float,
        duration_s: float = 3e-3,
        *,
        is_slab: bool = False,
        rephase: bool = True,
        time_bw_product: float = 4.0,
        axis: str = "z",
        pulse_type: str = "st",
        use: str = "excitation",
        passband_ripple: float = 0.01,
        stopband_ripple: float = 0.01,
    ) -> None:
        if thickness_m <= 0:
            raise ValueError("thickness_m must be positive")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf, gz, gz_reph = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            slice_thickness=thickness_m,
            time_bw_product=time_bw_product,
            pulse_type=pulse_type,
            passband_ripple=passband_ripple,
            stopband_ripple=stopband_ripple,
            return_gz=True,
            use=use,
            system=system,
        )
        gz.channel = axis
        gz_reph.channel = axis

        self.seq = pp.Sequence(system)

        if is_slab:
            if rephase:
                # Concatenated, not summed over one interval: the rephaser
                # starts where the selection lobe ends.
                gz_reph.delay = pp.calc_duration(gz)
                gz = pp.add_gradients(grads=[gz, gz_reph], system=system)
            self.seq.add_block(rf, gz)
        else:
            self.seq.add_block(rf, gz)
            if rephase:
                self.seq.add_block(gz_reph)

        self.center = rf_reference(rf)
