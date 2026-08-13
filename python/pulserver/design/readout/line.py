"""Cartesian line readouts: one frequency-encoded line per repetition."""

from __future__ import annotations

__all__ = ["LineReadout2D", "LineReadout3D"]

from typing import Any

import numpy as np

from ... import pypulseq as pp
from ..._core._module import SequenceModule
from ._common import (
    AXES,
    as_tuple,
    bridge,
    left_align_rephaser,
    present,
    solve_delay,
    solve_rephasing,
)

#: Fraction of ``max_grad`` the readout lobe may reach, leaving the rest for
#: the phase encodes riding alongside it.
_READOUT_GRAD_MARGIN = 0.8

_SPOILING_POSITIONS = ("pre", "post")


class _LineReadout(SequenceModule):
    """One Cartesian line, from the RF that starts it to the end of the TR.

    A readout module spans a whole repetition rather than only the acquisition
    window, because TE and TR are the numbers a caller actually has and
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

        seq.add_block(readout.gx_pre, pp.scale_grad(readout.gy_pre, ky))

    where ``ky`` runs over ``(index - n_y / 2) / (n_y / 2)``. Which lines are
    played, and in what order, is the loop's business.

    Attributes
    ----------
    rf : RfEvent
        The pulse the module was given.
    gz : GradEvent
        Its selection gradient, if one was given.
    gz_reph : GradEvent
        Its rephaser, if one was given, left-aligned in whichever block follows
        the pulse.
    gx_pre : GradEvent
        Readout prephaser, right-aligned in the prewinder block.
    gx : GradEvent
        Readout lobe.
    gx_spoil : GradEvent
        The lobe that closes the TR on the readout axis: a pure rewinder when
        ``spoiling_cycles`` is zero, a bridged spoiler otherwise.
    gy_pre, gz_pre : TrapEvent
        Phase encodes at their largest step, to be scaled per shot. ``gz_pre``
        is 3D only.
    gy_rew, gz_rew : TrapEvent
        The negated encodes that unwind them, for a balanced TR.
    gx_flyback : TrapEvent
        Rewinder played between echoes of a monopolar train.
    gx_rev : GradEvent
        The negated lobe that reads the even echoes of a bipolar train.
    adc : AdcEvent
        The acquisition window, shared by every echo.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``, in order. Absent when ``labels`` is empty,
        and a bare event rather than a list when there is one.
    wait_te, wait_tr : DelayEvent
        Present only when a TE or TR longer than the minimum was asked for.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The pulse that opens the repetition.
    gz : GradEvent, optional
        A selection gradient played in the same block as ``rf``. Pass an
        excitation's ``gz``, or nothing for a hard pulse.
    gz_reph : GradEvent, optional
        The rephaser that unwinds ``gz``. Pass a 2D excitation's ``gz_reph``
        and the readout carries it, left-aligned in the first block after the
        pulse so it runs straight off the selection lobe: the TE wait when
        there is one, the prewinder block otherwise. A slab excitation
        (``is_slab=True``) has already merged it into ``gz`` and passes
        nothing.
    fov : float or sequence of float
        Field of view (m), per encoded axis, readout first.
    matrix : int or sequence of int
        Matrix size, per encoded axis. This sets the gradient *areas*; how many
        lines are actually played is the scan loop's business.
    te : float, optional
        Echo time (s), from the RF isodelay to the first echo. ``None`` is as
        short as possible.
    tr : float, optional
        Repetition time (s), over the whole module. ``None`` is as short as
        possible.
    partial_echo : float, optional
        Fraction of the full echo acquired, in ``(0.5, 1]``. Truncates the
        samples *before* the echo, so it shortens TE.
    oversampling : float, optional
        Readout oversampling. Densifies the sampling -- ``delta_kx`` shrinks
        and the sampled field of view grows -- while the k-space width, and so
        the resolution, is fixed by ``fov`` and ``matrix`` alone.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth. Read ``bandwidth_hz`` for what was
        achieved: the dwell has to land on the ADC raster while the readout
        duration lands on the gradient raster, and both together rarely admit
        the number asked for.
    spoiling_cycles : float, optional
        Residual dephasing left at the end of the TR, in cycles across
        ``voxel_size_m``. Zero is balanced.
    voxel_size_m : float, optional
        Length the spoiling is counted over (m). Defaults to the readout
        resolution.
    spoiling_position : {'post', 'pre'}, optional
        Which side of the acquisition the dephasing lobe sits on.
    n_echoes : int, optional
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
        If a count or a fraction is out of range, or the requested TE or TR is
        shorter than the module can achieve.
    """

    #: 2 or 3. The only thing that separates the two shipped line readouts.
    _ndim = 2

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        gz_reph: Any = None,
        *,
        fov: Any,
        matrix: Any,
        te: float | None = None,
        tr: float | None = None,
        partial_echo: float = 1.0,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        spoiling_position: str = "post",
        n_echoes: int = 1,
        flyback: bool = True,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        ndim = self._ndim
        n_echoes = int(n_echoes)
        if n_echoes < 1:
            raise ValueError("n_echoes must be >= 1")
        if spoiling_position not in _SPOILING_POSITIONS:
            raise ValueError(
                f"spoiling_position must be one of {_SPOILING_POSITIONS}, got {spoiling_position!r}"
            )
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if not 0.5 < partial_echo <= 1.0:
            raise ValueError("partial_echo must be in (0.5, 1]")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")

        fov = as_tuple(fov, ndim, "fov")
        n = as_tuple(matrix, ndim, "matrix", int)
        if min(n) < 2 or min(fov) <= 0:
            raise ValueError("every matrix size must be >= 2 and every fov positive")

        # Sampling. Partial echo drops samples from before the echo, so the
        # full count sets where k starts and the acquired count sets the ADC.
        delta_kx = 1.0 / (oversampling * fov[0])
        n_full = round(oversampling * n[0])
        n_post = n_full // 2
        n_samples = max(n_post + 1, round(partial_echo * n_full))
        n_pre = n_samples - n_post
        readout_area = n_samples * delta_kx
        dwell, readout_duration = pp.calc_adc_timing(
            n_samples,
            1.0 / readout_bandwidth_hz,
            grad_raster_time=system.grad_raster_time,
            adc_raster_time=system.adc_raster_time,
            min_readout_duration=readout_area / (_READOUT_GRAD_MARGIN * system.max_grad),
        )

        if voxel_size_m is None:
            voxel_size_m = fov[0] / n[0]
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        spoil_area = spoiling_cycles / voxel_size_m

        # The readout lobe, and the two lobes that bracket it.
        gx = pp.make_trapezoid(
            channel="x", flat_area=readout_area, flat_time=readout_duration, system=system
        )
        amplitude, rise_time, fall_time = gx.amplitude, gx.rise_time, gx.fall_time
        pre_area = -n_pre * delta_kx
        post_area = -n_post * delta_kx

        if spoiling_position == "pre" and spoil_area:
            # Bridged into the readout lobe: the prewinder climbs to the
            # plateau itself, so its area *is* the k the flat top starts at and
            # the lobe keeps a ramp only on the far side.
            gx_pre = bridge(system, "x", pre_area - spoil_area, 0.0, amplitude)
            flat_top_start = 0.0
        else:
            # The lobe's own rise happens before the first sample and winds
            # half a ramp of k while it does. Uncompensated, that offsets the
            # whole line -- which a reconstruction sees as a first-order phase
            # rather than as an error, so nothing downstream would report it.
            gx_pre = pp.make_trapezoid(
                channel="x", area=pre_area - 0.5 * rise_time * amplitude, system=system
            )
            flat_top_start = rise_time

        if spoiling_position == "post" and spoil_area:
            gx_spoil = bridge(system, "x", post_area + spoil_area, amplitude, 0.0)
        else:
            gx_spoil = pp.make_trapezoid(
                channel="x", area=post_area - 0.5 * fall_time * amplitude, system=system
            )

        # A bridge already supplies the ramp on the side it joins.
        if spoil_area:
            gx = _reshape_readout(
                system, gx, spoiling_position == "pre", spoiling_position == "post"
            )

        adc = pp.make_adc(
            num_samples=n_samples, dwell=dwell, delay=flat_top_start, system=system
        )

        # Phase encoding, at its largest step: resolution alone fixes it, since
        # field of view and matrix cancel.
        gy_pre = pp.make_phase_encoding("y", fov[1] / n[1], system=system)
        gy_rew = pp.scale_grad(gy_pre, -1.0)
        if ndim == 3:
            gz_pre = pp.make_phase_encoding("z", fov[2] / n[2], system=system)
            gz_rew = pp.scale_grad(gz_pre, -1.0)

        # Monopolar spends a rewinder between echoes so every one is read in
        # the same direction; bipolar spends nothing and reads alternate echoes
        # backwards.
        if n_echoes > 1 and flyback:
            gx_flyback = pp.make_trapezoid(channel="x", area=-_area(gx), system=system)
        if n_echoes > 1 and not flyback:
            gx_rev = pp.scale_grad(gx, -1.0)

        # The prewinder block already carries one gradient per encoded axis, so
        # a rephaser sharing one of those channels has nowhere to sit.
        gz_reph = left_align_rephaser(gz_reph, AXES[:ndim], type(self).__name__)

        _prewinder = [gx_pre, gy_pre, gz_pre] if ndim == 3 else [gx_pre, gy_pre]
        if ndim == 3:
            gx_spoil, gy_rew, gz_rew = pp.align(left=[gx_spoil, gy_rew, gz_rew])
            _rewinder = [gx_spoil, gy_rew, gz_rew]
        else:
            gx_spoil, gy_rew = pp.align(left=[gx_spoil, gy_rew])
            _rewinder = [gx_spoil, gy_rew]

        adc_labels = [pp.make_label(type="SET", label=name, value=0) for name in labels or ()]

        self.seq = pp.Sequence(system)

        if gz is not None:
            self.seq.add_block(rf, gz)
        else:
            self.seq.add_block(rf)

        # `add_block` rounds a block up onto the block raster, so a span taken
        # from `calc_duration` alone is a raster short of where the block ends.
        rf_center = float(rf.delay) + float(rf.center)
        te_base = self.seq.duration()[0] - rf_center + flat_top_start + n_pre * dwell
        wait_span, pre_span, echo_time = solve_rephasing(
            te,
            te_base,
            pp.calc_duration(*_prewinder, *present(trigger)),
            pp.calc_duration(gz_reph) if gz_reph is not None else 0.0,
            system,
        )

        # Align only now that the span is settled, since the delay alignment
        # writes is part of the event the caller receives. The delay sets the
        # span everything is right-aligned against and is then dropped: the
        # aligned gradients already end there, so they hold the block open.
        _prewinder.append(pp.make_delay(pre_span))
        if trigger is not None:
            trigger, *_prewinder = pp.align(left=[trigger], right=_prewinder)
        else:
            _prewinder = pp.align(right=_prewinder)
        _prewinder = _prewinder[:-1]
        if ndim == 3:
            gx_pre, gy_pre, gz_pre = _prewinder
        else:
            gx_pre, gy_pre = _prewinder

        if wait_span:
            wait_te = pp.make_delay(wait_span)
            if gz_reph is not None:
                self.seq.add_block(wait_te, gz_reph)
            else:
                self.seq.add_block(wait_te)

        _pre_block = [*_prewinder, *present(None if wait_span else gz_reph), *present(trigger)]
        self.seq.add_block(*_pre_block)

        for i_echo in range(n_echoes):
            if n_echoes > 1 and flyback and i_echo:
                self.seq.add_block(gx_flyback)
            lobe = gx if (flyback or i_echo % 2 == 0) else gx_rev
            self.seq.add_block(lobe, adc, *adc_labels)

        self.seq.add_block(*_rewinder)

        tr_min = self.seq.duration()[0]
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            self.seq.add_block(wait_tr)

        self.echo_time = echo_time
        self.center = echo_time + rf_center
        self.bandwidth_hz = 1.0 / dwell
        self.n_samples = n_samples
        self.delta_kx = delta_kx
        self.readout_duration = readout_duration


def _area(event) -> float:
    """Zeroth moment of a gradient event, whichever kind it is."""
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


def _reshape_readout(system, gx, bridged_start: bool, bridged_end: bool):
    """Drop whichever ramp a bridged spoiler has already provided."""
    amplitude = gx.amplitude
    spans = [
        0.0 if bridged_start else gx.rise_time,
        gx.flat_time,
        0.0 if bridged_end else gx.fall_time,
    ]
    amplitudes = [
        amplitude if bridged_start else 0.0,
        amplitude,
        amplitude,
        amplitude if bridged_end else 0.0,
    ]
    times = np.cumsum([0.0, *spans])
    keep = [0, *(index + 1 for index, span in enumerate(spans) if span > 0)]
    return pp.make_extended_trapezoid(
        channel="x",
        amplitudes=np.asarray(amplitudes, dtype=float)[keep],
        times=times[keep],
        system=system,
    )


class LineReadout2D(_LineReadout):
    """One Cartesian line, frequency-encoded along x and phase-encoded along y.

    ``fov`` and ``matrix`` take two values here, readout first. See
    :class:`_LineReadout` for the timing, spoiling and echo-train arguments.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.LineReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=128,
    ... )
    >>> int(readout.adc.num_samples)
    128

    The rephaser rides the prewinder block, at its head:

    >>> readout.blocks[1] == (readout.gx_pre, readout.gy_pre, readout.gz_reph)
    True
    """

    _ndim = 2


class LineReadout3D(_LineReadout):
    """One Cartesian line of a 3D slab, phase-encoded along y and z.

    ``fov`` and ``matrix`` take three values, readout first. See
    :class:`_LineReadout` for the shared arguments.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> slab = design.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True)
    >>> readout = design.LineReadout3D(
    ...     system, slab.rf, slab.gz,
    ...     fov=(0.22, 0.22, 0.12), matrix=(128, 128, 64),
    ... )
    >>> readout.gy_pre.channel, readout.gz_pre.channel
    ('y', 'z')
    """

    _ndim = 3
