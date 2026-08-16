"""Excitation and refocusing of one slice or one slab."""

from __future__ import annotations

__all__ = ["SpatialSelectiveExcitation", "SpatialSelectiveRefocusing"]

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

        # The thickness the pulse and its selection gradient actually produce,
        # which is the pulse's measured bandwidth over the gradient it is
        # played on. It differs from `thickness_m` by however much the
        # designed envelope's spectrum differs from its nominal time-bandwidth
        # product, so it is what a reconstruction should be told. Taken before
        # a slab's rephaser is concatenated onto the lobe.
        # Half a hertz, because the flanks are found by interpolating the
        # spectrum and the default 10 Hz grid moves the answer by 0.7%. This
        # runs once per module, so the resolution costs nothing that matters.
        self.slice_thickness = float(pp.calc_rf_bandwidth(rf, dw=0.5) / abs(gz.amplitude))

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


class SpatialSelectiveRefocusing(RfModule):
    """An SLR 180 under a selection gradient, with its crushers bridged on.

    The refocusing half of a slice-selective spin echo. The SLR pulse is
    designed with the ``"se"`` profile rather than the excitation one, because
    the two solve different problems: an excitation profile is the transverse
    magnetisation a pulse produces from equilibrium, a refocusing profile is
    how completely it inverts, and a pulse optimal for one is not optimal for
    the other.

    The crushers ride the selection lobe rather than waiting for it to fall to
    zero. A pair of identical lobes, one each side of the pulse, is invisible
    to the refocused pathway -- the phase one winds the other unwinds -- and
    fatal to the FID an imperfect 180 leaves behind, which sees only the second
    lobe. Bridging them onto the plateau makes the whole thing a single
    gradient event, which is also what lets a readout accept it as its ``gz``.

    The pulse is phased a quarter turn from the excitation, the CPMG
    condition.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    thickness_m : float
        Slice thickness (m).
    flip_angle_deg : float, optional
        Nominal flip angle (degrees).
    duration_s : float, optional
        Pulse duration (s).
    spoiling_cycles : float, optional
        Cycles of dephasing each crusher winds across ``voxel_size_m``. Zero
        leaves the bare selection lobe.
    voxel_size_m : float, optional
        Length the dephasing is counted over (m).
    time_bw_product : float, optional
        Time-bandwidth product.
    axis : {'z', 'x', 'y'}, optional
        Selection axis; the crushers share it.
    phase_offset_rad : float, optional
        RF phase. The CPMG quarter turn by default.
    use : str, optional
        What the pulse is for; the trajectory core negates accumulated k at a
        refocusing pulse.
    passband_ripple, stopband_ripple : float, optional
        Ripple allowed in each band of the slice profile.

    Attributes
    ----------
    rf_ref : RfEvent
        The refocusing pulse, delayed onto the plateau between the crushers.
    gz : GradEvent
        Crusher, selection plateau and crusher, as one gradient.

    Raises
    ------
    ValueError
        If a thickness, duration or voxel size is not positive, ``axis`` is not
        a gradient channel, or ``spoiling_cycles`` is negative.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> refocusing = design.SpatialSelectiveRefocusing(pp.Opts(), 5e-3)
    >>> len(refocusing.blocks), refocusing.gz.type
    (1, 'grad')

    A spin echo is then the same readout an excitation would open, given the
    refocusing pulse instead::

        readout = design.LineReadout2D(
            system, refocusing.rf_ref, refocusing.gz,
            fov=0.22, matrix=128, te=15e-3,
        )
    """

    def init_module(
        self,
        system: pp.Opts,
        thickness_m: float,
        flip_angle_deg: float = 180.0,
        duration_s: float = 3e-3,
        *,
        spoiling_cycles: float = 4.0,
        voxel_size_m: float = 1e-3,
        time_bw_product: float = 4.0,
        axis: str = "z",
        phase_offset_rad: float = np.pi / 2,
        use: str = "refocusing",
        passband_ripple: float = 0.01,
        stopband_ripple: float = 0.01,
    ) -> None:
        if thickness_m <= 0:
            raise ValueError("thickness_m must be positive")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")

        rf_ref, gz, _ = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=duration_s,
            slice_thickness=thickness_m,
            time_bw_product=time_bw_product,
            pulse_type="se",
            phase_offset=phase_offset_rad,
            passband_ripple=passband_ripple,
            stopband_ripple=stopband_ripple,
            return_gz=True,
            use=use,
            system=system,
        )
        gz.channel = axis

        if spoiling_cycles:
            # Each crusher is solved to arrive at, or leave from, the plateau,
            # so the three pieces share their vertices and become one waveform:
            # crusher up, selection flat top, crusher down.
            _, times_pre, amplitudes_pre = pp.make_crusher(
                spoiling_cycles, voxel_size_m, axis, grad_end=gz.amplitude, system=system
            )
            _, times_post, amplitudes_post = pp.make_crusher(
                spoiling_cycles, voxel_size_m, axis, grad_start=gz.amplitude, system=system
            )
            flat_start = float(times_pre[-1])
            flat_end = flat_start + float(gz.flat_time)
            rf_ref.delay += flat_start - float(gz.rise_time)
            gz = pp.make_extended_trapezoid(
                channel=axis,
                amplitudes=np.concatenate(
                    [amplitudes_pre, [gz.amplitude], amplitudes_post[1:]]
                ),
                times=np.concatenate(
                    [times_pre, [flat_end], flat_end + np.asarray(times_post[1:])]
                ),
                system=system,
            )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf_ref, gz)

        self.center = rf_reference(rf_ref)
