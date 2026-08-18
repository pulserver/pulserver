"""Balanced SSFP readouts: a fully rewound repetition with the echo at TR/2."""

from __future__ import annotations

__all__ = ["BssfpReadout2D", "BssfpReadout3D"]

from typing import Any

import numpy as np

from ... import pypulseq as pp
from ..._core._module import SequenceModule
from ._common import as_tuple, bridge, present, solve_delay

#: Fraction of ``max_grad`` the readout plateau may reach.
_READOUT_GRAD_MARGIN = 0.8


class _BssfpReadout(SequenceModule):
    """One balanced repetition -- rewind, excite, read -- with TE at TR/2.

    Every axis returns to k = 0 between one pulse centre and the next, so the
    rephasers are built here rather than taken: their areas follow from that,
    not from the pulse. Pass the excitation with ``rephase=False``.

    A scan loop, with the alternating phase and the two transients a steady
    state needs::

        nominal = bssfp.rf.amplitude
        bssfp.rf.amplitude = 0.5 * nominal          # the half-flip pulse, and
        seq.add_block(bssfp.rf, bssfp.gz, *bssfp.prep_labels)
        seq.add_block(bssfp.wait_prep, *bssfp.train_labels)
        bssfp.rf.amplitude = nominal                # half a TR before the first

        for shot, ky in enumerate(plan):
            bssfp.rf.phase_offset = bssfp.adc.phase_offset = np.pi * ((shot + 1) % 2)
            if shot:                                # the first has no plateau to
                seq.add_block(bssfp.gx_rew,         # leave and nothing to rewind
                              pp.scale_grad(bssfp.gy_rew, plan[shot - 1]), bssfp.gz_rew)
            else:
                seq.add_block(bssfp.wait_rewind, bssfp.gz_rew)
            seq.add_block(bssfp.rf, bssfp.gz)
            seq.add_block(bssfp.gx, bssfp.adc, pp.scale_grad(bssfp.gy_pre, ky),
                          bssfp.gz_pre)
        seq.add_block(bssfp.gx_rew, pp.scale_grad(bssfp.gy_rew, plan[-1]), bssfp.gz_rew,
                      *bssfp.end_labels)

    See :doc:`../reference/design` for the timing that fixes TE at TR/2, why
    the half-flip pulse opposes the first excitation, and what ``ONCE`` marks.

    Attributes
    ----------
    rf : RfEvent
        The excitation, replayed every repetition with the phase and amplitude
        the loop sets.
    gz : GradEvent
        Its selection gradient, if one was given.
    gx_rew, gx : GradEvent
        The read lobe that closes a repetition and the one that opens the next.
        ``gx`` is prephaser and plateau in a single waveform.
    gy_pre, gy_rew : TrapEvent
        In-plane encode at its largest step, and the same negated, to be scaled
        per shot.
    gz_pre, gz_rew : TrapEvent
        The slice rephasers, one each side of the pulse. Absent for a hard
        pulse.
    gz_partition, gz_partition_rew : TrapEvent
        Partition encode at its largest step, and the same negated, each
        sharing a rephaser's window. 3D only; add one to a rephaser rather than
        playing it alone.
    adc : AdcEvent
        The acquisition window, spanning the readout plateau.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``; a bare event when there is one.
    prep_labels, train_labels, end_labels : list of LabelSetEvent
        ``ONCE`` set to 1, 0 and 2, marking the entry transient, the steady
        state and the ramp-down.
    wait_prep : DelayEvent
        What separates the half-flip pulse from the first excitation, so that
        interval is half a repetition. Absent under ``half_flip_prep=False``.
    wait_rewind : DelayEvent
        Stands in for the rewind block of the first repetition.
    tr, te : float
        Repetition and echo time (s); ``te`` is always half of ``tr``.
    bandwidth_hz : float
        Achieved receiver bandwidth.
    n_samples : int
        Samples per repetition.
    delta_kx : float
        Read-axis k-space step (1/m).
    readout_duration : float
        Sampling window (s).

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The excitation.
    gz : GradEvent, optional
        Its selection gradient, played in the excitation block. Nothing for a
        hard pulse. Pass an excitation built with ``rephase=False``: the
        rephasers belong to the balance condition and are built here.
    fov : float or sequence of float
        Field of view (m), per encoded axis, readout first.
    matrix : int or sequence of int
        Matrix size, per encoded axis.
    tr : float, optional
        Repetition time (s). ``None`` is as short as possible; a longer one is
        padded evenly either side of the echo, so TE stays at TR/2.
    half_flip_prep : bool, optional
        Publish ``wait_prep``, which needs half the acquisition window to be at
        least the excitation block's tail. Turn it off to ramp the train in
        with dummy repetitions instead.
    oversampling : float, optional
        Read oversampling.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth. Read ``bandwidth_hz`` for what the two
        rasters allowed.
    labels : sequence of str, optional
        Counters emitted on the acquisition block.
    trigger : event, optional
        A trigger or digital output armed on the rewind block.

    Raises
    ------
    ValueError
        If a count is out of range, the acquisition window is shorter than
        twice the excitation block's tail, or the requested TR is shorter than
        the module can achieve.
    """

    #: 2 or 3. The only thing that separates the two shipped bSSFP readouts.
    _ndim = 2

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        *,
        fov: Any,
        matrix: Any,
        tr: float | None = None,
        half_flip_prep: bool = True,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        ndim = self._ndim
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")

        fov = as_tuple(fov, ndim, "fov")
        n = as_tuple(matrix, ndim, "matrix", int)
        if min(n) < 2 or min(fov) <= 0:
            raise ValueError("every matrix size must be >= 2 and every fov positive")

        raster = system.block_duration_raster

        # A balanced readout is symmetric by construction -- the echo sits in
        # the middle of the plateau -- so there is no partial echo to ask for.
        delta_kx = 1.0 / (oversampling * fov[0])
        n_samples = round(oversampling * n[0])
        readout_area = n_samples * delta_kx
        dwell, readout_duration = pp.calc_adc_timing(
            n_samples,
            1.0 / readout_bandwidth_hz,
            grad_raster_time=system.grad_raster_time,
            adc_raster_time=system.adc_raster_time,
            min_readout_duration=readout_area
            / (_READOUT_GRAD_MARGIN * system.max_grad),
        )
        shape = pp.make_trapezoid(
            channel="x",
            flat_area=readout_area,
            flat_time=readout_duration,
            system=system,
        )
        amplitude = float(shape.amplitude)
        # The plateau has to reach the end of its block, or the lobe that
        # follows has no plateau to leave from -- and the block runs on past
        # the last sample by the receiver's dead time.
        flat_span = pp.ceil_to_raster(readout_duration + system.adc_dead_time, raster)

        # What the read axis does outside the acquisition: down off the
        # plateau and back onto it, carrying the echo to k = 0 on the way.
        entry_area = -0.5 * amplitude * readout_duration
        gx_rew = bridge(
            system,
            "x",
            -amplitude * (flat_span - 0.5 * readout_duration),
            amplitude,
            0.0,
        )
        entry_times, entry_amplitudes = _vertices(system, entry_area, 0.0, amplitude)

        rf_center = float(rf.delay) + float(rf.center)
        excite_span = _span(raster, rf, gz)
        prep_span = (
            0.5 * (flat_span + (flat_span - readout_duration)) + rf_center - excite_span
        )
        if half_flip_prep and prep_span < -1e-12:
            raise ValueError(
                f"the {flat_span * 1e3:.3f} ms acquisition block is too short beside the "
                f"{(excite_span - rf_center) * 1e3:.3f} ms that follows the pulse centre in "
                f"the excitation block, so no half-flip pulse can sit half a repetition before "
                f"the first excitation; lower readout_bandwidth_hz, shorten the pulse, or pass "
                f"half_flip_prep=False and ramp the train in with dummy repetitions"
            )

        # The rephasers are a pair that cancels the selection lobe outright.
        # Split so the one after the pulse unwinds what followed the pulse
        # centre, which is the half that has to be unwound by the echo.
        selection_area = _area(gz)
        post_area = -0.5 * selection_area
        pre_area = -(selection_area + post_area)

        # Every z lobe is built on the same ramp, which is what makes a
        # rephaser and a partition encode add up to a single trapezoid instead
        # of an arbitrary waveform.
        ramp = pp.ceil_to_raster(
            system.max_grad / system.max_slew, system.grad_raster_time
        )
        step_z = 0.5 * n[2] / fov[2] if ndim == 3 else 0.0
        floor = max(
            _z_floor(system, max(abs(pre_area), abs(post_area)) + step_z, ramp, raster),
            _span(raster, gx_rew, trigger),
            _span(raster, pp.make_phase_encoding("y", fov[1] / n[1], system=system)),
            pp.ceil_to_raster(float(entry_times[-1]), raster),
        )

        # TE = TR/2 puts the echo as far after the pulse centre as the next
        # pulse centre is after the echo, which is one equation between the two
        # windows either side of the excitation block.
        offset = (flat_span - readout_duration) + 2.0 * rf_center - excite_span
        rewind_span = pp.ceil_to_raster(max(floor, floor - offset), raster)
        tr_min = 2.0 * rewind_span + offset + flat_span + excite_span
        rewind_span += pp.round_to_raster(
            0.5 * solve_delay(tr, tr_min, "TR", system), raster
        )
        encode_span = rewind_span + offset

        gy_pre = pp.make_phase_encoding(
            "y", fov[1] / n[1], system=system, duration=encode_span
        )
        gy_rew = pp.scale_grad(
            pp.make_phase_encoding(
                "y", fov[1] / n[1], system=system, duration=rewind_span
            ),
            -1.0,
        )
        gz_pre = _lobe(system, post_area, encode_span, ramp)
        gz_rew = _lobe(system, pre_area, rewind_span, ramp)
        if ndim == 3:
            gz_partition = _lobe(system, step_z, encode_span, ramp)
            gz_partition_rew = _lobe(system, -step_z, rewind_span, ramp)
            self.register(gz_partition=gz_partition, gz_partition_rew=gz_partition_rew)

        gx = pp.make_extended_trapezoid(
            channel="x",
            times=np.append(entry_times, entry_times[-1] + flat_span),
            amplitudes=np.append(entry_amplitudes, amplitude),
            system=system,
        )
        gx.delay = encode_span - float(entry_times[-1])
        adc = pp.make_adc(
            num_samples=n_samples, dwell=dwell, delay=encode_span, system=system
        )

        if half_flip_prep:
            wait_prep = pp.make_delay(prep_span)
        wait_rewind = pp.make_delay(rewind_span)

        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]
        prep_labels = [pp.make_label(type="SET", label="ONCE", value=1)]
        train_labels = [pp.make_label(type="SET", label="ONCE", value=0)]
        end_labels = [pp.make_label(type="SET", label="ONCE", value=2)]

        # One repetition of the centre partition, laid out from the pulse so
        # that it starts and ends with every axis at rest.
        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, *present(gz))
        self.seq.add_block(gx, adc, gy_pre, *present(gz_pre), *adc_labels)
        self.seq.add_block(gx_rew, gy_rew, *present(gz_rew), *present(trigger))
        self.register(
            wait_rewind=wait_rewind,
            prep_labels=prep_labels,
            train_labels=train_labels,
            end_labels=end_labels,
            **({"wait_prep": wait_prep} if half_flip_prep else {}),
        )

        self.tr = rewind_span + excite_span + encode_span + flat_span
        self.te = 0.5 * self.tr
        self.center = excite_span + encode_span + 0.5 * readout_duration
        self.bandwidth_hz = 1.0 / dwell
        self.n_samples = n_samples
        # Balanced and never a partial echo, so k crosses zero at the midpoint.
        self.center_sample = n_samples // 2
        self.delta_kx = delta_kx
        self.readout_duration = readout_duration


def _area(event: Any) -> float:
    """Zeroth moment of a gradient event, or zero when there is none."""
    if event is None:
        return 0.0
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


def _span(raster: float, *events: Any) -> float:
    """Where a block holding ``events`` ends, ignoring the ones that are None."""
    events = [event for event in events if event is not None]
    return pp.ceil_to_raster(pp.calc_duration(*events), raster) if events else 0.0


def _vertices(system: pp.Opts, area: float, grad_start: float, grad_end: float):
    """A bridged lobe as the arrays it is built from, so it can be extended."""
    _, times, amplitudes = pp.make_extended_trapezoid_area(
        area=area, channel="x", grad_start=grad_start, grad_end=grad_end, system=system
    )
    return np.asarray(times, dtype=float), np.asarray(amplitudes, dtype=float)


def _lobe(system: pp.Opts, area: float, duration: float, ramp: float):
    """A z trapezoid of ``area`` over ``duration``, or ``None`` if there is none.

    The ramp is given rather than solved, so that two lobes over the same
    window share their vertex times whatever their areas -- which is what lets
    a rephaser and a partition encode be added into one trapezoid.
    """
    if abs(area) < 1e-12:
        return None
    return pp.make_trapezoid(
        channel="z", area=area, duration=duration, rise_time=ramp, system=system
    )


def _z_floor(system: pp.Opts, area: float, ramp: float, raster: float) -> float:
    """Shortest window that holds a z trapezoid of ``area`` on fixed ramps.

    Sized against the largest lobe a shot can ask for -- a rephaser with the
    outermost partition encode added to it -- because those two are summed into
    one trapezoid and it is their sum that has to stay inside ``max_grad``.
    """
    return pp.ceil_to_raster(max(2.0 * ramp, ramp + area / system.max_grad), raster)


class BssfpReadout2D(_BssfpReadout):
    """A slice-selective balanced SSFP repetition, phase-encoded along y.

    ``fov`` and ``matrix`` take two values, readout first. See
    :class:`_BssfpReadout` for the arguments and the scan loop.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 40.0, 5e-3, 1e-3, rephase=False)
    >>> bssfp = design.BssfpReadout2D(
    ...     system, excitation.rf, excitation.gz, fov=0.28, matrix=192,
    ... )
    >>> bssfp.te == bssfp.tr / 2
    True
    """

    _ndim = 2


class BssfpReadout3D(_BssfpReadout):
    """A slab-selective balanced SSFP repetition, encoded along y and z.

    ``fov`` and ``matrix`` take three values, readout first. The partition
    encode shares the rephasers' window and is added onto them, which stays one
    trapezoid because the two are built over the same timing -- so a partition
    costs an amplitude, not a waveform::

        seq.add_block(bssfp.gx, bssfp.adc, pp.scale_grad(bssfp.gy_pre, ky),
                      pp.add_gradients([bssfp.gz_pre,
                                        pp.scale_grad(bssfp.gz_partition, kz)],
                                       system=system))

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> slab = design.SpatialSelectiveExcitation(system, 40.0, 0.12, 1e-3, rephase=False)
    >>> bssfp = design.BssfpReadout3D(
    ...     system, slab.rf, slab.gz, fov=(0.28, 0.28, 0.12), matrix=(192, 192, 48),
    ... )
    >>> pp.add_gradients(
    ...     [bssfp.gz_pre, pp.scale_grad(bssfp.gz_partition, 1.0)], system=system
    ... ).type
    'trap'
    """

    _ndim = 3
