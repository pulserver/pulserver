"""Echo-planar readouts: one blipped, ramp-sampled train per repetition."""

from __future__ import annotations

__all__ = ["EpiReadout2D", "EpiReadout3D"]

from typing import Any

import numpy as np

from ... import pypulseq as pp
from ..._core._module import SequenceModule
from ._common import (
    as_tuple,
    left_align_rephaser,
    present,
    solve_delay,
    solve_rephasing,
)

#: Fraction of ``max_grad`` the readout plateau may reach.
_READOUT_GRAD_MARGIN = 0.8


class _EpiReadout(SequenceModule):
    """One excitation and the EPI train that follows it: ``etl`` lines per TR.

    The train walks a fixed pattern of phase-encode steps and replays it every
    repetition. Only the *origin* moves, so the scan loop scales one prewinder
    per axis and leaves the blips alone::

        for shot, (ky, kz) in enumerate(plan):
            seq.add_block(epi.rf, epi.gz)
            epi.shot_labels[0].value, epi.shot_labels[1].value = ky, kz
            seq.add_block(epi.gx_pre, pp.scale_grad(epi.gy_pre, ky / half_ny),
                          *epi.shot_labels)
            for line in range(epi.etl):
                seq.add_block(epi.gx[line], epi.adc, *epi.line_labels[line])
            seq.add_block(epi.gx_spoil)

    The blips ride the readout ramps rather than costing time of their own,
    which is what keeps the echo spacing near the sampling window. See
    :doc:`../reference/design` for the built-in orderings and what each is for.

    Attributes
    ----------
    rf : RfEvent
        The excitation.
    gz : GradEvent
        Its selection gradient, if one was given.
    gz_reph : GradEvent
        Its rephaser, if one was given.
    gx_pre : TrapEvent
        Read prewinder, half a line lobe against the first line's polarity.
    gx : list of GradEvent
        One read lobe per line: alternating polarity for a blipped train, the
        same lobe every time for a flyback one.
    gx_flyback : TrapEvent
        The rewind between two lines. ``flyback`` only.
    gy_blips, gz_blips : list
        One phase-encode event per line, or ``None`` where no step is asked
        for, always ``etl`` long. A blipped train's entry carries the second
        half of the step into the line and the first half of the next out of
        it; a flyback train's is the whole step, played in the rewind block
        *after* that line, so its last entry is always ``None``.
        ``gz_blips`` is 3D only.
    gy_pre, gz_pre : TrapEvent
        Phase encodes at their largest step, to be scaled per shot. ``gz_pre``
        is 3D only.
    gy_rew, gz_rew : TrapEvent
        The encodes negated, in the block that closes the repetition.
    gx_spoil : GradEvent
        The lobe that closes the read axis, bridged off the last line.
    adc : AdcEvent
        The acquisition window, shared by every line.
    shot_labels : tuple of LabelEvent
        ``SET`` counters on the prewinder block, one per name in ``labels``.
        The loop writes the shot's origin into them. Always a tuple, however
        many names there are, because the loop splats it.
    line_labels : tuple of tuple of LabelEvent
        Per line, the ``INC`` counters that step from the previous line to this
        one -- the ordering, expressed as labels. Empty for the first line,
        which the ``SET`` already placed.
    order : numpy.ndarray
        ``(etl, 2)`` integer ``(ky, kz)`` offsets from the shot's origin.
    wait_te : DelayEvent
        Present only when a TE longer than the minimum was asked for.
    wait_tr : DelayEvent
        Present only when a TR longer than the minimum was asked for.
    etl : int
        Lines per repetition.
    esp : float
        Echo spacing (s).
    echo_times : numpy.ndarray
        Each line's echo time (s) from the excitation isodelay.
    echo_time : float
        The first line's, which is what ``te`` sets.
    bandwidth_hz : float
        Achieved receiver bandwidth.
    n_samples : int
        Samples per line.
    delta_kx : float
        Read-axis k-space step (1/m).
    blip_span : float
        Time one phase-encode step takes (s). It is paid out of the read ramps.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The pulse that opens the repetition. A refocusing pulse builds the
        readout half of a spin-echo EPI.
    gz : GradEvent, optional
        A selection gradient played in the same block as ``rf``.
    gz_reph : GradEvent, optional
        The rephaser that unwinds ``gz``.
    fov : float or sequence of float
        Field of view (m), per encoded axis, readout first.
    matrix : int or sequence of int
        Matrix size, per encoded axis.
    order : array_like, optional
        ``(etl,)`` or ``(etl, 2)`` integer offsets from the shot's origin, one
        row per line. Supplying one silences the generator arguments below; the
        default asks :func:`~pulserver.pypulseq.calc_epi_order` for a train.
    etl : int, optional
        Lines per repetition. Defaults to what one shot of the requested
        scheme needs to cross the phase-encode matrix.
    scheme : {'linear', 'caipi', 'zigzag'}, optional
        Which built-in ordering to generate. See
        :func:`~pulserver.pypulseq.calc_epi_order`.
    acceleration, segments, partition_acceleration, caipi_shift, extent
        Passed to :func:`~pulserver.pypulseq.calc_epi_order`.
    te : float, optional
        Excitation isodelay to the **first** echo (s). ``None`` is as short as
        possible; every other echo follows at ``esp`` intervals, and
        ``echo_times`` lists them.
    tr : float, optional
        Repetition time (s), over the whole module.
    oversampling : float, optional
        Read oversampling.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth. Read ``bandwidth_hz`` for what the two
        rasters allowed.
    ramp_sampling : bool, optional
        Sample across the read ramps as well as the plateau, which is what
        makes the echo spacing short. Turn it off for a rectangular window at
        the cost of a longer train.
    flyback : bool, optional
        Read every line in the same direction, rewinding between them, instead
        of alternating polarity. It costs a rewind per line and buys away the
        odd-even inconsistency that a bipolar train leaves behind.
    spoiling_cycles : float, optional
        Read-axis spoiling at the end of the repetition, in cycles across
        ``voxel_size_m``.
    voxel_size_m : float, optional
        Length the spoiling is counted over (m). The read resolution by
        default.
    labels : sequence of str, optional
        Counters for the encoded axes, phase encode first. Two names for a 3D
        train, one for 2D.
    trigger : event, optional
        A trigger or digital output armed on the prewinder block.

    Raises
    ------
    ValueError
        If a count is out of range, ``order`` is the wrong shape or steps
        outside the matrix, ``labels`` does not name one counter per encoded
        axis, or the requested TE or TR is shorter than the train can achieve.
    """

    #: 2 or 3. The only thing that separates the two shipped EPI readouts.
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
        order: Any = None,
        etl: int | None = None,
        scheme: str = "linear",
        acceleration: int = 1,
        segments: int = 1,
        partition_acceleration: int = 1,
        caipi_shift: int = 1,
        extent: int | None = None,
        te: float | None = None,
        tr: float | None = None,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 500e3,
        ramp_sampling: bool = True,
        flyback: bool = False,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        ndim = self._ndim
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")

        fov = as_tuple(fov, ndim, "fov")
        n = as_tuple(matrix, ndim, "matrix", int)
        if min(n) < 2 or min(fov) <= 0:
            raise ValueError("every matrix size must be >= 2 and every fov positive")

        if order is None:
            if etl is None:
                etl = -(-int(n[1]) // (int(acceleration) * int(segments)))
            order = pp.calc_epi_order(
                etl,
                scheme=scheme,
                acceleration=acceleration,
                segments=segments,
                partition_acceleration=partition_acceleration,
                caipi_shift=caipi_shift,
                extent=extent,
            )
        order = _checked_order(order, n, ndim)
        etl = len(order)

        labels = tuple(labels or ())
        if labels and len(labels) != ndim - 1:
            raise ValueError(
                f"labels names the counter for each encoded axis after the readout, so a "
                f"{ndim}D train takes {ndim - 1}, got {len(labels)}"
            )

        raster = system.block_duration_raster
        delta_k = tuple(1.0 / value for value in fov)

        # One blip window serves every step, so it is built for the largest
        # step in the pattern and every other is that shape scaled. Half of it
        # runs into a line and half out, so it has to be an even raster count.
        steps = np.diff(order, axis=0)
        blip_span = 0.0
        for axis in range(1, ndim):
            widest = int(np.max(np.abs(steps[:, axis - 1]))) if len(steps) else 0
            if widest:
                trial = pp.make_trapezoid(
                    channel="yz"[axis - 1], area=widest * delta_k[axis], system=system
                )
                blip_span = max(blip_span, pp.calc_duration(trial))
        blip_span = pp.ceil_to_raster(blip_span, 2.0 * system.grad_raster_time)

        # Sampling.
        n_full = round(oversampling * n[0])
        k_width = n_full * delta_k[0]
        dwell, sample_span = pp.calc_adc_timing(
            n_full,
            1.0 / readout_bandwidth_hz,
            grad_raster_time=system.grad_raster_time,
            adc_raster_time=system.adc_raster_time,
            min_readout_duration=k_width / (_READOUT_GRAD_MARGIN * system.max_grad),
        )
        # A flyback line reads on its own and hands the step to the rewind, so
        # nothing rides its ramps and the pad is only what the receiver needs.
        lobe, echo_offset = _read_lobe(
            system, k_width, sample_span, 0.0 if flyback else blip_span, ramp_sampling
        )
        adc = pp.make_adc(
            num_samples=n_full,
            dwell=dwell,
            delay=echo_offset - 0.5 * sample_span,
            system=system,
        )

        # The prewinder is half a lobe against the first line, so the window is
        # centred on k = 0 whatever shape the ramps took.
        gx_pre = pp.make_trapezoid(channel="x", area=-0.5 * _area(lobe), system=system)
        if flyback:
            # Every line reads the same way and a rewind between them puts the
            # read axis back where it started, carrying the step as it goes.
            gap_span = pp.ceil_to_raster(
                max(
                    pp.calc_duration(
                        pp.make_trapezoid(channel="x", area=-_area(lobe), system=system)
                    ),
                    blip_span,
                ),
                system.grad_raster_time,
            )
            gx_flyback = pp.make_trapezoid(
                channel="x", area=-_area(lobe), duration=gap_span, system=system
            )
            gx = [lobe] * etl
            esp = pp.calc_duration(lobe) + gap_span
            gy_blips = _gap_blips(system, "y", steps[:, 0], delta_k[1], gap_span, etl)
            gz_blips = (
                _gap_blips(system, "z", steps[:, 1], delta_k[2], gap_span, etl)
                if ndim == 3
                else [None] * etl
            )
        else:
            gx = [
                pp.scale_grad(lobe, 1.0 if line % 2 == 0 else -1.0)
                for line in range(etl)
            ]
            esp = pp.calc_duration(lobe)
            gy_blips = _blip_events(
                system, "y", steps[:, 0], delta_k[1], blip_span, lobe, etl
            )
            gz_blips = (
                _blip_events(system, "z", steps[:, 1], delta_k[2], blip_span, lobe, etl)
                if ndim == 3
                else [None] * etl
            )

        gy_pre = pp.make_phase_encoding("y", fov[1] / n[1], system=system)
        gz_pre = (
            pp.make_phase_encoding("z", fov[2] / n[2], system=system)
            if ndim == 3
            else None
        )
        pre_span = pp.ceil_to_raster(
            max(
                pp.calc_duration(event) for event in (gx_pre, gy_pre, *present(gz_pre))
            ),
            raster,
        )
        gx_pre = pp.make_trapezoid(
            channel="x", area=-0.5 * _area(lobe), duration=pre_span, system=system
        )
        gy_pre = pp.make_phase_encoding(
            "y", fov[1] / n[1], system=system, duration=pre_span
        )
        if ndim == 3:
            gz_pre = pp.make_phase_encoding(
                "z", fov[2] / n[2], system=system, duration=pre_span
            )
        gy_rew = pp.scale_grad(gy_pre, -1.0)
        gz_rew = pp.scale_grad(gz_pre, -1.0) if ndim == 3 else None

        # The train ends half a lobe from the centre, on whichever side the
        # last line's polarity left it. The closing lobe travels back through
        # k = 0 and keeps going by whatever spoiling was asked for, so it never
        # degenerates to nothing however the two are balanced.
        if voxel_size_m is None:
            voxel_size_m = fov[0] / n[0]
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        gx_spoil = pp.make_trapezoid(
            channel="x",
            area=-float(np.sign(_area(gx[-1])))
            * (0.5 * abs(_area(lobe)) + spoiling_cycles / voxel_size_m),
            system=system,
        )

        gz_reph = left_align_rephaser(
            gz_reph, ("x", "y", "z"[: ndim - 2]), type(self).__name__
        )
        shot_labels = tuple(
            pp.make_label(type="SET", label=name, value=0) for name in labels
        )
        line_labels = tuple(
            tuple(
                pp.make_label(type="INC", label=name, value=int(steps[line - 1, axis]))
                for axis, name in enumerate(labels)
            )
            if line
            else ()
            for line in range(etl)
        )

        # Timing. The rephaser goes in whichever block follows the pulse, so it
        # runs straight off the selection lobe.
        exc_span = _span(system, rf, gz)
        rf_center = float(rf.delay) + float(rf.center)
        wait, pre_span, echo_time = solve_rephasing(
            te,
            (exc_span - rf_center) + echo_offset,
            pre_span,
            _span(system, gz_reph),
            system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, *present(gz))
        if wait:
            wait_te = pp.make_delay(wait)
            self.seq.add_block(wait_te, *present(gz_reph))
        prewinder = [gx_pre, gy_pre, *present(gz_pre)]
        if not wait:
            prewinder += present(gz_reph)
        if pre_span > pp.calc_duration(*prewinder) + 1e-12:
            prewinder.append(pp.make_delay(pre_span))
        self.seq.add_block(*prewinder, *shot_labels, *present(trigger))
        for line in range(etl):
            if flyback:
                self.seq.add_block(gx[line], adc, *line_labels[line])
                if line < etl - 1:
                    self.seq.add_block(
                        gx_flyback, *present(gy_blips[line]), *present(gz_blips[line])
                    )
            else:
                self.seq.add_block(
                    gx[line],
                    adc,
                    *present(gy_blips[line]),
                    *present(gz_blips[line]),
                    *line_labels[line],
                )
        self.seq.add_block(gx_spoil, gy_rew, *present(gz_rew))

        tr_min = self.seq.duration()[0]
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            self.seq.add_block(wait_tr)

        self.register(
            gx=gx,
            gy_blips=gy_blips,
            shot_labels=shot_labels,
            line_labels=line_labels,
            order=order,
        )
        if ndim == 3:
            self.register(gz_blips=gz_blips)
        self.center = exc_span + wait + pre_span + echo_offset
        self.echo_time = echo_time
        self.echo_times = echo_time + esp * np.arange(etl)
        self.etl = etl
        self.esp = esp
        self.blip_span = blip_span
        self.bandwidth_hz = 1.0 / dwell
        self.n_samples = n_full
        self.delta_kx = delta_k[0]


def _checked_order(order: Any, n: tuple[int, ...], ndim: int) -> np.ndarray:
    """The ordering as ``(etl, 2)`` integer offsets, read-only."""
    values = np.asarray(order)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 1:
        raise ValueError("order must have shape (etl,) or (etl, 2)")
    if values.shape[1] == 1:
        values = np.column_stack((values[:, 0], np.zeros(len(values))))
    if values.shape[1] != 2:
        raise ValueError("order must have shape (etl,) or (etl, 2)")
    if not np.all(values == np.rint(values)):
        raise ValueError("order must contain integers: they index phase-encode lines")
    values = np.rint(values).astype(np.intp)
    if ndim == 2 and np.any(values[:, 1] != 0):
        raise ValueError(
            "order steps the partition axis, which a 2D train does not encode; use "
            "EpiReadout3D, whose second column is the partition offset"
        )
    for axis in range(1, ndim):
        span = int(np.ptp(values[:, axis - 1])) if len(values) > 1 else 0
        if span >= n[axis]:
            raise ValueError(
                f"order spans {span + 1} steps on the {'yz'[axis - 1]} axis, which does not "
                f"fit the {n[axis]} it is encoded over"
            )
    values.setflags(write=False)
    return values


def _read_lobe(
    system: pp.Opts,
    k_width: float,
    sample_span: float,
    blip_span: float,
    ramp_sampling: bool,
):
    """One line's read gradient, and how far into it the echo falls.

    The lobe is padded symmetrically either side of the acquisition window, by
    enough to hold half a phase-encode step and never less than the receiver's
    own dead time. A step therefore runs entirely outside the samples it
    separates, and the window stays centred -- which is what lets the prewinder
    be half a lobe whatever shape the ramps took.
    """
    pad = pp.ceil_to_raster(
        max(0.5 * blip_span, system.adc_dead_time), system.grad_raster_time
    )
    if ramp_sampling:
        # Over-drive the area so that, once the parts outside the window are
        # discounted, what remains inside it is still the full sweep -- then
        # scale, which lowers the amplitude and so cannot break a limit. The
        # first guess bounds the discount by the ramps; a pad wider than a ramp
        # eats into the plateau too, so the guess is checked and raised.
        total = sample_span + 2.0 * pad
        requested = k_width + pad**2 * system.max_slew
        for _ in range(8):
            wide = pp.make_trapezoid(
                channel="x", area=requested, duration=total, system=system
            )
            inside = _area_between(wide, pad, pad + sample_span)
            if inside >= k_width:
                break
            requested *= 1.05 * k_width / inside
        else:
            raise ValueError(
                "could not fit the readout sweep into its sampling window; lower "
                "readout_bandwidth_hz or oversampling"
            )
        return pp.scale_grad(wide, k_width / inside), 0.5 * total

    plateau = pp.make_trapezoid(
        channel="x", flat_area=k_width, flat_time=sample_span, system=system
    )
    rise = pp.ceil_to_raster(
        max(float(plateau.rise_time) + 0.5 * blip_span, system.adc_dead_time),
        system.grad_raster_time,
    )
    amplitude = float(plateau.amplitude)
    lobe = pp.make_extended_trapezoid(
        channel="x",
        times=np.array([0.0, rise, rise + sample_span, 2.0 * rise + sample_span]),
        amplitudes=np.array([0.0, amplitude, amplitude, 0.0]),
        system=system,
    )
    return lobe, rise + 0.5 * sample_span


def _area_between(trapezoid: Any, start: float, stop: float) -> float:
    """Zeroth moment of a trapezoid over one interval of it.

    Exact rather than a closed form for the corners, because the pad either
    side of the window is not always shorter than the ramp it sits on.
    """
    rise, flat = float(trapezoid.rise_time), float(trapezoid.flat_time)
    fall, amplitude = float(trapezoid.fall_time), float(trapezoid.amplitude)
    times = np.array([0.0, rise, rise + flat, rise + flat + fall])
    amplitudes = np.array([0.0, amplitude, amplitude, 0.0])
    edges = np.unique(
        np.concatenate((times[(times > start) & (times < stop)], [start, stop]))
    )
    return float(np.trapezoid(np.interp(edges, times, amplitudes), edges))


def _blip_events(
    system: pp.Opts,
    channel: str,
    steps: np.ndarray,
    delta_k: float,
    blip_span: float,
    anchor: Any,
    etl: int,
) -> list:
    """One phase-encode event per line, split across the read ramps.

    A step between two lines is halved: the first half plays out on the falling
    ramp of the line it leaves, the second on the rising ramp of the line it
    enters. Every step is the same shape scaled, so they all take ``blip_span``
    however far they move and the echo spacing stays constant.
    """
    widest = int(np.max(np.abs(steps))) if len(steps) else 0
    if not widest or blip_span == 0.0:
        return [None] * etl
    template = pp.make_trapezoid(
        channel=channel, area=widest * delta_k, duration=blip_span, system=system
    )
    halves = [
        pp.split_gradient_at(
            grad=pp.scale_grad(template, float(step) / widest),
            time_point=0.5 * blip_span,
            system=system,
        )
        if step
        else None
        for step in steps
    ]
    events = []
    for line in range(etl):
        leaving = (
            halves[line][0] if line < etl - 1 and halves[line] is not None else None
        )
        entering = (
            halves[line - 1][1] if line and halves[line - 1] is not None else None
        )
        events.append(_flank(system, entering, leaving, anchor))
    return events


def _gap_blips(
    system: pp.Opts,
    channel: str,
    steps: np.ndarray,
    delta_k: float,
    gap_span: float,
    etl: int,
) -> list:
    """One whole phase-encode step per rewind block, for a flyback train.

    There is no ramp to hide on, so the step is played whole in the gap the
    rewind already occupies. Entry ``j`` belongs to the gap after line ``j``,
    which is why the last is always ``None``.
    """
    widest = int(np.max(np.abs(steps))) if len(steps) else 0
    if not widest:
        return [None] * etl
    template = pp.make_trapezoid(
        channel=channel, area=widest * delta_k, duration=gap_span, system=system
    )
    events = [
        pp.scale_grad(template, float(step) / widest) if step else None
        for step in steps
    ]
    return [*events, None]


def _flank(system: pp.Opts, entering: Any, leaving: Any, anchor: Any):
    """The two half-steps flanking one line, as the single event that line plays."""
    if entering is None and leaving is None:
        return None
    if entering is None:
        return pp.align(right=[leaving], left=[anchor])[0]
    if leaving is None:
        return pp.align(left=[entering, anchor])[0]
    right, left, _ = pp.align(right=[leaving], left=[entering, anchor])
    return pp.add_gradients([left, right], system=system)


def _span(system: pp.Opts, *events: Any) -> float:
    """Where a block holding ``events`` ends, ignoring the ones that are None."""
    events = [event for event in events if event is not None]
    if not events:
        return 0.0
    return pp.ceil_to_raster(pp.calc_duration(*events), system.block_duration_raster)


def _area(event: Any) -> float:
    """Zeroth moment of a gradient event, whichever kind it is."""
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


class EpiReadout2D(_EpiReadout):
    """A single- or multi-shot EPI train, frequency-encoded along x.

    ``fov`` and ``matrix`` take two values, readout first. See
    :class:`_EpiReadout` for the ordering, timing and spoiling arguments.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> excitation = design.SpatialSelectiveExcitation(system, 60.0, 3e-3)
    >>> epi = design.EpiReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=64, labels=("LIN",),
    ... )
    >>> epi.etl, int(epi.adc.num_samples)
    (64, 64)

    Segmenting shortens the train and widens the blip, leaving the lattice
    alone:

    >>> half = design.EpiReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=64, segments=2,
    ... )
    >>> half.etl, int(half.order[1, 0])
    (32, 2)

    The whole plane from one excitation: the prewinder places k at a corner
    and every blip steps it one line, so the train is the lattice.

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import trajectory

       system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
       excitation = design.SpatialSelectiveExcitation(system, 60.0, 3e-3)
       epi = design.EpiReadout2D(
           system, excitation.rf, excitation.gz, excitation.gz_reph,
           fov=0.22, matrix=32,
       )
       trajectory(
           epi,
           ky=[-1.0],
           per="shot",
           label="line",
           title="EpiReadout2D, a 32-line train",
       )

    The blips are what separates the lines, and the alternating read lobe is
    why every other one is reversed:

    .. plot::
       :include-source:

       import pulserver.design as design
       import pulserver.pypulseq as pp

       system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
       excitation = design.SpatialSelectiveExcitation(system, 60.0, 3e-3)
       epi = design.EpiReadout2D(
           system, excitation.rf, excitation.gz, excitation.gz_reph,
           fov=0.22, matrix=32,
       )
       epi.plot(time_disp="ms", grad_disp="mT/m", stacked=True, plot_now=False)
    """

    _ndim = 2


class EpiReadout3D(_EpiReadout):
    """A slab-selective EPI train, phase-encoded along y and partition-encoded along z.

    ``fov`` and ``matrix`` take three values, readout first. The partition axis
    is blipped line by line, so a whole CAIPI pattern is played inside one
    train.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> slab = design.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
    >>> epi = design.EpiReadout3D(
    ...     system, slab.rf, slab.gz, fov=(0.22, 0.22, 0.12), matrix=(64, 64, 32),
    ...     scheme="caipi", acceleration=2, segments=2, partition_acceleration=3,
    ...     labels=("LIN", "PAR"),
    ... )
    >>> epi.etl
    16

    The partition walks the CAIPI cycle and wraps, so its blips take two values:

    >>> sorted(set(int(step) for step in epi.order[1:, 1] - epi.order[:-1, 1]))
    [-1, 2]

    Under ``scheme="caipi"`` the partition blips walk a cycle alongside the
    phase blips, so the train covers a sheared lattice rather than one plane:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import trajectory

       system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
       slab = design.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
       epi = design.EpiReadout3D(
           system, slab.rf, slab.gz,
           fov=(0.22, 0.22, 0.12), matrix=(32, 32, 8),
           scheme="caipi", acceleration=2, partition_acceleration=2, caipi_shift=1,
       )
       trajectory(
           epi,
           ky=[-1.0],
           kz=[-1.0],
           per="shot",
           plane="yz",
           label="line",
           title="EpiReadout3D, the CAIPI lattice one train covers",
       )
    """

    _ndim = 3
