"""Cartesian line readouts: one frequency-encoded line per repetition."""

from __future__ import annotations

__all__ = ["LineReadout2D", "LineReadout3D"]

from typing import Any

import numpy as np

from ... import pypulseq as pp
from ..._core._module import SequenceModule
from ._common import (
    AXES,
    DEFAULT_BANDWIDTH_HZ,
    as_tuple,
    bridge,
    readout_sampling,
    solve_delay,
)

_SPOILING_POSITIONS = ("pre", "post")


class _LineReadout(SequenceModule):
    """One Cartesian line, from the RF that starts it to the end of the TR.

    A readout module spans a whole repetition rather than only the acquisition
    window, because TE and TR are the numbers a caller actually has, and
    neither can be budgeted without the pulse. The pulse is passed in as an
    **event**, not as a module, so the same class serves an excitation --
    giving a gradient echo -- or a refocusing pulse, giving the second half of
    a spin echo, where TE is then measured from the refocusing isodelay.

    Three regimes, from two numbers:

    ============================ ================================================
    ``spoiling_cycles = 0``      balanced: every axis is rewound to k = 0
    ``> 0``, ``position='post'`` SSFP-FID: the dephasing lobe follows the readout
    ``> 0``, ``position='pre'``  SSFP-Echo: it precedes the readout instead
    ============================ ================================================

    Both spoiled regimes leave the same residual moment across the TR; they
    differ in which side of the acquisition it sits on, and so in which
    coherence pathway is read. The spoiler is *bridged* -- it rides straight
    off the readout lobe rather than waiting for it to fall to zero.

    A dephasing lobe placed *before* the readout necessarily offsets this
    repetition's own k-space by the residual: what such a sequence acquires is
    not this excitation's FID but the echo refocused from the previous one.
    :meth:`calculate_kspace` traces the FID from rest, so under
    ``spoiling_position='pre'`` it reports a trajectory that never crosses
    k = 0. That is the pathway distinction, not a design error; ``echo_time``
    remains the interval it says it is.

    Phase encoding is designed at full amplitude and left there. A scan loop
    scales it per shot::

        seq.add_block(pp.scale_grad(readout.gy_phase, ky_scale), readout.gx_pre)

    where ``ky_scale`` runs over ``(index - matrix_y / 2) / (matrix_y / 2)``.
    Which lines are played, and in what order, is the loop's business.

    Attributes
    ----------
    rf : RfEvent
        The pulse the module was given.
    gz_select : GradEvent
        Its selection gradient, if one was given.
    gx_pre : GradEvent
        Readout prephaser, right-aligned in the prewinder block.
    gx_read : GradEvent
        Readout lobe, one per echo. A list when ``num_echoes > 1``.
    gx_rew : GradEvent
        Readout rewinder or bridged spoiler, left-aligned after the last echo.
    gy_phase : TrapEvent
        In-plane phase encode at its largest step, to be scaled per shot.
    gz_phase : TrapEvent
        Partition encode at its largest step. 3D only.
    gy_rew, gz_rew : TrapEvent
        The negated encodes that unwind them, for a balanced TR.
    adc : AdcEvent
        The acquisition window, shared by every echo.
    adc_labels : list of LabelSetEvent
        One per name in ``labels``, in order; empty when ``labels`` is.
    gx_flyback : TrapEvent
        Rewinder played between echoes of a monopolar train.
    gx_read_reversed : GradEvent
        The negated lobe that reads the even echoes of a bipolar train.
    wait_te, wait_tr : DelayEvent
        Present only when a TE or TR longer than the minimum was asked for.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The pulse that opens the repetition.
    gz_select : GradEvent, optional
        A selection gradient played in the same block as ``rf``. Pass an
        excitation's merged ``gz_slab``, or nothing for a hard pulse.
    fov_m : float or sequence of float
        Field of view (m), per encoded axis.
    matrix : int or sequence of int
        Matrix size, per encoded axis. This sets the gradient *areas*; how
        many lines are actually played is the scan loop's business.
    te : float, optional
        Echo time (s), from the RF isodelay to the first echo. ``None`` is as
        short as possible.
    tr : float, optional
        Repetition time (s), over the whole module. ``None`` is as short as
        possible.
    partial_echo : float, optional
        Fraction of the full echo acquired, in ``(0.5, 1]``.
    oversampling : float, optional
        Readout oversampling factor.
    bandwidth_hz : float, optional
        Requested receiver bandwidth (Hz); read ``adc.dwell`` for what was
        achieved.
    spoiling_cycles : float, optional
        Residual dephasing left at the end of the TR, in cycles across
        ``voxel_size_m``. Zero is balanced.
    voxel_size_m : float, optional
        Length the spoiling is counted over (m). Defaults to the readout
        resolution.
    spoiling_position : {'post', 'pre'}, optional
        Which side of the acquisition the dephasing lobe sits on.
    num_echoes : int, optional
        Echoes per repetition.
    flyback : bool, optional
        With more than one echo: rewind between echoes so every one is read in
        the same direction (monopolar, the default), or alternate the readout
        sign (bipolar), which is faster but reads even echoes backwards and
        puts any gradient-delay error into a phase difference between them.
    labels : sequence of str, optional
        Counters emitted on the acquisition block. The loop writes the values.
    trigger : event, optional
        A trigger or digital output armed on the prewinder block.

    Raises
    ------
    ValueError
        If ``num_echoes`` is below 1, ``spoiling_position`` is not ``'pre'`` or
        ``'post'``, an axis is not a gradient channel, or the requested TE or
        TR is shorter than the module can achieve.
    """

    #: Encoded axes, readout first. ``_LineReadout`` is 2D or 3D by this alone.
    _ndim = 2

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz_select: Any = None,
        *,
        fov_m: Any,
        matrix: Any,
        te: float | None = None,
        tr: float | None = None,
        partial_echo: float = 1.0,
        oversampling: float = 1.0,
        bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        spoiling_position: str = "post",
        num_echoes: int = 1,
        flyback: bool = True,
        axes: tuple[str, ...] | None = None,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        ndim = self._ndim
        num_echoes = int(num_echoes)
        if num_echoes < 1:
            raise ValueError("num_echoes must be >= 1")
        if spoiling_position not in _SPOILING_POSITIONS:
            raise ValueError(
                f"spoiling_position must be one of {_SPOILING_POSITIONS}, got {spoiling_position!r}"
            )
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        axes = tuple(axes) if axes is not None else AXES[:ndim]
        if len(axes) != ndim or len(set(axes)) != ndim or any(axis not in AXES for axis in axes):
            raise ValueError(f"axes must be {ndim} distinct gradient channels, got {axes!r}")

        fov = as_tuple(fov_m, ndim, "fov_m")
        size = as_tuple(matrix, ndim, "matrix", int)
        read_axis, *phase_axes = axes

        sampling = readout_sampling(
            system,
            size[0],
            fov[0],
            oversampling=oversampling,
            partial_echo=partial_echo,
            bandwidth_hz=bandwidth_hz,
        )
        if voxel_size_m is None:
            voxel_size_m = fov[0] / size[0]
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        residual = spoiling_cycles / voxel_size_m

        # --- the readout lobe, and the two lobes that bracket it -----------
        gx_read = pp.make_trapezoid(
            channel=read_axis,
            flat_area=sampling.k_width,
            flat_time=sampling.duration,
            system=system,
        )
        amplitude = gx_read.amplitude
        # Where the readout starts and ends in k, relative to the echo.
        pre_area = -sampling.num_pre * sampling.delta_k
        post_area = -sampling.num_post * sampling.delta_k

        spoil_pre = residual if spoiling_position == "pre" else 0.0
        spoil_post = residual if spoiling_position == "post" else 0.0

        if spoil_pre:
            # Bridged into the readout lobe: the prewinder climbs to the
            # plateau itself, so its area *is* the k the flat top starts at and
            # the lobe keeps a ramp only on the far side.
            gx_pre = bridge(system, read_axis, pre_area - spoil_pre, 0.0, amplitude)
        else:
            # The lobe's own rise happens before the first sample, and winds
            # half a ramp of k while it does. Uncompensated, that offsets the
            # whole line -- which is a first-order phase in the image, not
            # something a reconstruction notices as an error.
            gx_pre = pp.make_trapezoid(
                channel=read_axis,
                area=pre_area - 0.5 * gx_read.rise_time * amplitude,
                system=system,
            )
        if spoil_post:
            gx_rew = bridge(system, read_axis, post_area + spoil_post, amplitude, 0.0)
        else:
            gx_rew = pp.make_trapezoid(
                channel=read_axis,
                area=post_area - 0.5 * gx_read.fall_time * amplitude,
                system=system,
            )

        # Bridging removes the ramp the lobe would otherwise need on that side,
        # so a pre-bridged lobe opens straight onto its plateau.
        flat_top_start = 0.0 if spoil_pre else gx_read.rise_time
        gx_read = _reshape_readout(system, read_axis, gx_read, bool(spoil_pre), bool(spoil_post))

        adc = pp.make_adc(
            num_samples=sampling.num_samples,
            dwell=sampling.dwell,
            delay=flat_top_start,
            system=system,
        )

        # --- phase encoding, at its largest step ---------------------------
        phase_encodes = [
            pp.make_phase_encoding(axis, fov[n + 1] / size[n + 1], system=system)
            for n, axis in enumerate(phase_axes)
        ]
        phase_rewinds = [pp.scale_grad(event, -1.0) for event in phase_encodes]

        # --- the echo train -------------------------------------------------
        # Monopolar spends a rewinder between echoes so every one is read in
        # the same direction; bipolar spends nothing and reads alternate
        # echoes backwards, which is faster but puts any gradient-delay error
        # into a phase difference between them.
        gx_read_reversed = None
        gx_flyback = None
        if num_echoes > 1:
            if flyback:
                gx_flyback = pp.make_trapezoid(
                    channel=read_axis, area=-_area(gx_read), system=system
                )
            else:
                gx_read_reversed = pp.scale_grad(gx_read, -1.0)

        # --- align the prewinder and the rewinder --------------------------
        # Align before anything is added: alignment writes a delay, and the
        # delay is part of the event a caller receives.
        _prewinder = pp.align(right=[gx_pre, *phase_encodes])
        if trigger is not None:
            trigger, *_prewinder = pp.align(left=[trigger], right=list(_prewinder))
        gx_pre, *_encodes = _prewinder
        gx_rew, *_rewinds = pp.align(left=[gx_rew, *phase_rewinds])
        for _axis, _encode, _rewind in zip(phase_axes, _encodes, _rewinds, strict=True):
            self.register(**{f"g{_axis}_phase": _encode, f"g{_axis}_rew": _rewind})

        adc_labels = [pp.make_label(type="SET", label=name, value=0) for name in labels or ()]
        # Registered rather than left to the automatic path, so one label is a
        # one-entry list and the loop indexes it the same way regardless.
        self.register(adc_labels=adc_labels)

        # --- lay out the repetition ----------------------------------------
        self.seq = pp.Sequence(system)
        if gz_select is not None:
            self.seq.add_block(rf, gz_select)
        else:
            self.seq.add_block(rf)

        _rf_reference = float(rf.delay) + float(rf.center)
        _echo_offset = flat_top_start + sampling.echo_offset
        # The prewinder span is rounded onto the block raster because that is
        # what `add_block` will do to it, and a TE computed from an unrounded
        # span would be a raster short of where the echo actually lands.
        _prewinder_span = pp.ceil_to_raster(
            pp.calc_duration(gx_pre, *_encodes), system.block_duration_raster
        )
        te_min = self.seq.duration()[0] - _rf_reference + _prewinder_span + _echo_offset
        _te_delay = solve_delay(te, te_min, "TE", system)
        if _te_delay:
            wait_te = pp.make_delay(_te_delay)
            self.seq.add_block(wait_te)

        if trigger is not None:
            self.seq.add_block(gx_pre, *_encodes, trigger)
        else:
            self.seq.add_block(gx_pre, *_encodes)

        for _echo in range(num_echoes):
            if gx_flyback is not None and _echo:
                self.seq.add_block(gx_flyback)
            _lobe = gx_read if (gx_read_reversed is None or _echo % 2 == 0) else gx_read_reversed
            self.seq.add_block(_lobe, adc, *adc_labels)

        self.seq.add_block(gx_rew, *_rewinds)

        _tr_delay = solve_delay(tr, self.seq.duration()[0], "TR", system)
        if _tr_delay:
            wait_tr = pp.make_delay(_tr_delay)
            self.seq.add_block(wait_tr)

        self.echo_time = te_min + _te_delay
        self.center = self.echo_time + _rf_reference
        self.bandwidth_hz = 1.0 / sampling.dwell
        self.sampling = sampling


def _reshape_readout(system, channel, gx_read, bridged_start: bool, bridged_end: bool):
    """Drop whichever ramps a bridged spoiler has already provided."""
    if not (bridged_start or bridged_end):
        return gx_read
    amplitude = gx_read.amplitude
    spans = [
        0.0 if bridged_start else gx_read.rise_time,
        gx_read.flat_time,
        0.0 if bridged_end else gx_read.fall_time,
    ]
    amplitudes = [
        0.0 if not bridged_start else amplitude,
        amplitude,
        amplitude,
        0.0 if not bridged_end else amplitude,
    ]
    times = np.cumsum([0.0, *spans])
    keep = [0, *(i + 1 for i, span in enumerate(spans) if span > 0)]
    return pp.make_extended_trapezoid(
        channel=channel,
        amplitudes=np.asarray(amplitudes, dtype=float)[keep],
        times=times[keep],
        system=system,
    )


def _area(event) -> float:
    """Zeroth moment of a gradient event, whichever kind it is."""
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


class LineReadout2D(_LineReadout):
    """One Cartesian line, frequency-encoded along x and phase-encoded along y.

    See :class:`_LineReadout` for the timing, spoiling and echo-train
    arguments, which are shared. ``fov_m`` and ``matrix`` take two values here,
    readout first.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.LineReadout2D(
    ...     system, excitation.rf, excitation.gz_select,
    ...     fov_m=0.22, matrix=128,
    ... )
    >>> int(readout.adc.num_samples)
    128
    """

    _ndim = 2


class LineReadout3D(_LineReadout):
    """One Cartesian line of a 3D slab, phase-encoded along y and z.

    See :class:`_LineReadout` for the shared arguments. ``fov_m`` and
    ``matrix`` take three values, readout first.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> slab = design.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True)
    >>> readout = design.LineReadout3D(
    ...     system, slab.rf, slab.gz_slab,
    ...     fov_m=(0.22, 0.22, 0.12), matrix=(128, 128, 64),
    ... )
    >>> readout.gy_phase.channel, readout.gz_phase.channel
    ('y', 'z')
    """

    _ndim = 3
