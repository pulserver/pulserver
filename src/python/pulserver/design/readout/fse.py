"""Fast spin-echo readouts: one CPMG echo train per repetition."""

from __future__ import annotations

__all__ = ["FseReadout2D", "FseReadout3D"]

from typing import Any

import numpy as np

from ... import pypulseq as pp
from .._module import SequenceModule
from ._common import (
    as_tuple,
    bridge,
    left_align_rephaser,
    present,
    solve_delay,
    wave_gradients,
)

#: Fraction of ``max_grad`` the readout plateau may reach.
_READOUT_GRAD_MARGIN = 0.8


class _FseReadout(SequenceModule):
    """One excitation and the CPMG train that follows it: ``etl`` lines per TR.

    Both pulses are passed in as events, so the same class builds a
    slice-selective 2D train, a slab-selective 3D train, or the single-slab 3D
    train that refocuses non-selectively. Every echo replays the same events;
    the flip schedule and the line each echo encodes are the scan loop's, moved
    on the published events::

        nominal = fse.rf_ref.amplitude
        for echo, (ky, flip) in enumerate(plan):
            fse.rf_ref.amplitude = nominal * flip / np.pi
            seq.add_block(fse.rf_ref, fse.gz_ref)
            if echo == 0 and fse.esp_first > fse.esp:
                seq.add_block(fse.wait_esp1)
            seq.add_block(fse.gx_bridge_pre, pp.scale_grad(fse.gy_pre, ky))
            seq.add_block(fse.gx, fse.adc)
            seq.add_block(fse.gx_bridge_post, pp.scale_grad(fse.gy_rew, ky))

    See :doc:`../reference/design` for why the encodes are separate trapezoids,
    what the read-axis crushing is for, and when ``esp_first`` exceeds ``esp``.

    Attributes
    ----------
    rf : RfEvent
        The excitation.
    gz : GradEvent
        Its selection gradient, if one was given.
    gz_reph : GradEvent
        Its rephaser, if one was given, left-aligned in the block that carries
        the prephaser so it runs straight off the selection lobe.
    rf_ref : RfEvent
        The refocusing pulse, one event for the whole train.
    gz_ref : GradEvent
        Its selection gradient and crushers, if one was given.
    gx_pre : TrapEvent
        Read prephaser, played once before the train and right-aligned in its
        block.
    gx_bridge_pre, gx_bridge_post : GradEvent
        The lobes that carry the read axis onto and off the readout plateau,
        one each side of every acquisition.
    gx : GradEvent
        The readout plateau, flat from end to end.
    gy_pre, gz_pre : TrapEvent
        Phase encodes at their largest step, to be scaled per echo. ``gz_pre``
        is 3D only.
    gy_rew, gz_rew : TrapEvent
        The encodes negated, unwinding each line before the next refocusing
        pulse.
    adc : AdcEvent
        The acquisition window, shared by every echo.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``; a bare event when there is one.
    wait_esp1 : DelayEvent
        Played after the first refocusing pulse only, when ``esp_first``
        exceeds ``esp``. Absent otherwise.
    wait_tr : DelayEvent
        Present only when a TR longer than the minimum was asked for.
    esp : float
        Echo spacing (s).
    esp_first : float
        Spacing of the first echo (s), equal to ``esp`` unless the excitation
        needed more room.
    echo_times : numpy.ndarray
        Each echo's time (s) from the excitation isodelay. Which of them is the
        effective echo time depends on the ordering the loop plays, so the
        module states them all and names none.
    etl : int
        Echoes per repetition.
    bandwidth_hz : float
        Achieved receiver bandwidth.
    n_samples : int
        Samples per echo.
    delta_kx : float
        Read-axis k-space step (1/m).
    readout_duration : float
        Sampling window (s).

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The excitation that opens the repetition.
    gz : GradEvent, optional
        A selection gradient played in the same block as ``rf``.
    gz_reph : GradEvent, optional
        The rephaser that unwinds ``gz``. A slab excitation
        (``is_slab=True``) has already merged it into ``gz`` and passes
        nothing.
    rf_ref : RfEvent
        The refocusing pulse, built with ``use="refocusing"``. Pass
        :class:`~pulserver.design.SpatialSelectiveRefocusing`'s ``rf_ref`` for
        a selective train, or
        :class:`~pulserver.design.NonSelectiveRefocusing`'s for the hard-pulse
        train that makes single-slab 3D FSE efficient.
    gz_ref : GradEvent, optional
        Selection gradient for ``rf_ref``, crushers included -- one gradient,
        because it is played in the pulse's own block.
        :class:`~pulserver.design.SpatialSelectiveRefocusing` publishes exactly
        that as its ``gz``. A non-selective train passes nothing and crushes on
        the read axis instead.
    fov : float or sequence of float
        Field of view (m), per encoded axis, readout first.
    matrix : int or sequence of int
        Matrix size, per encoded axis. This sets the gradient *areas*; how many
        lines are played, and in what order, is the scan loop's business.
    etl : int, optional
        Echo train length: refocusing pulses, and encoded lines, per
        repetition.
    esp : float, optional
        Echo spacing (s). ``None`` is as short as possible.
    esp_first : float, optional
        Spacing of the first echo (s). ``None`` is the shortest that
        accommodates the excitation, which is ``esp`` whenever the excitation
        fits in half of one.
    tr : float, optional
        Repetition time (s), over the whole module. ``None`` is as short as
        possible.
    partial_echo : float, optional
        Fraction of the full echo acquired, in ``(0.5, 1]``. Truncates the
        samples before the echo.
    oversampling : float, optional
        Read oversampling: ``delta_kx`` shrinks and the sampled field of view
        grows, while resolution is fixed by ``fov`` and ``matrix``.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth. Read ``bandwidth_hz`` for what the two
        rasters allowed.
    spoiling_cycles : float, optional
        Read-axis crushing each side of every acquisition, in cycles across
        ``voxel_size_m``.
    voxel_size_m : float, optional
        Length the crushing is counted over (m). The read resolution by
        default.
    labels : sequence of str, optional
        Counters emitted on the acquisition block. The loop writes the values.
    trigger : event, optional
        A trigger or digital output armed on the prephaser block.

    Raises
    ------
    ValueError
        If a count or fraction is out of range, ``rf_ref`` is not marked as a
        refocusing pulse, or the requested echo spacing or TR is shorter than
        the train can achieve.
    """

    #: 2 or 3. The only thing that separates the two shipped FSE readouts.
    _ndim = 2

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        gz_reph: Any = None,
        *,
        rf_ref: Any,
        gz_ref: Any = None,
        fov: Any,
        matrix: Any,
        etl: int = 8,
        esp: float | None = None,
        esp_first: float | None = None,
        tr: float | None = None,
        partial_echo: float = 1.0,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
        wave: str | None = None,
        wave_cycles: int = 8,
        wave_amplitude: float = 8e-3,
    ) -> None:
        ndim = self._ndim
        if wave is not None and ndim != 3:
            raise ValueError(
                "wave encoding spreads a voxel along the two encoded axes "
                "transverse to the readout, and a 2D readout has only one of "
                "them: the other is the slice"
            )
        etl = int(etl)
        if etl < 1:
            raise ValueError("etl must be >= 1")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if not 0.5 < partial_echo <= 1.0:
            raise ValueError("partial_echo must be in (0.5, 1]")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")
        if rf_ref.use != "refocusing":
            raise ValueError(
                f"rf_ref must be built with use='refocusing', got {rf_ref.use!r}: the "
                f"trajectory core negates the accumulated moment at a refocusing pulse, "
                f"and a train whose pulses are not marked as such analyses as a "
                f"trajectory that is not the one played"
            )

        fov = as_tuple(fov, ndim, "fov")
        n = as_tuple(matrix, ndim, "matrix", int)
        if min(n) < 2 or min(fov) <= 0:
            raise ValueError("every matrix size must be >= 2 and every fov positive")

        raster = system.block_duration_raster

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
            min_readout_duration=readout_area
            / (_READOUT_GRAD_MARGIN * system.max_grad),
        )

        if voxel_size_m is None:
            voxel_size_m = fov[0] / n[0]
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        spoil_area = spoiling_cycles / voxel_size_m

        # The readout plateau has no ramps of its own: both are carried by the
        # lobes that bridge it across the refocusing pulse.
        shape = pp.make_trapezoid(
            channel="x",
            flat_area=readout_area,
            flat_time=readout_duration,
            system=system,
        )
        amplitude = float(shape.amplitude)
        ramp_area = 0.5 * float(shape.rise_time) * amplitude
        adc = pp.make_adc(num_samples=n_samples, dwell=dwell, system=system)

        # The corkscrew rides under the plateau the samples are taken on. Both
        # encoded axes are free there -- the encodes are spent in the bridges
        # either side -- and each event is self-balanced, so an echo that
        # scales one to zero changes nothing else about the readout.
        gy_wave = gz_wave = None
        wave_peak = 0.0
        if wave is not None:
            built = wave_gradients(
                system,
                flat_time=readout_duration,
                delay=float(adc.delay),
                cycles=wave_cycles,
                amplitude=wave_amplitude,
                mode=wave,
            )
            gy_wave, gz_wave = built.get("y"), built.get("z")
            wave_peak = built["amplitude"]
        echo_offset = float(adc.delay) + n_pre * dwell
        flat_span = pp.ceil_to_raster(float(adc.delay) + readout_duration, raster)
        gx = pp.make_extended_trapezoid(
            channel="x",
            times=np.array([0.0, flat_span]),
            amplitudes=np.array([amplitude, amplitude]),
            system=system,
        )

        # k has to reach the same place before every acquisition, and the
        # refocusing pulse negates it in between, so the two lobes differ only
        # by how far the echo sits from the middle of the plateau.
        rebalance = amplitude * (0.5 * flat_span - echo_offset)
        gx_pre = pp.make_trapezoid(
            channel="x",
            area=ramp_area + spoil_area + 0.5 * amplitude * flat_span,
            system=system,
        )
        gx_bridge_pre = bridge(
            system, "x", ramp_area + spoil_area + rebalance, 0.0, amplitude
        )
        _, post_times, post_amplitudes = pp.make_extended_trapezoid_area(
            area=ramp_area + spoil_area - rebalance,
            channel="x",
            grad_start=amplitude,
            grad_end=0.0,
            system=system,
        )

        gz_reph = left_align_rephaser(gz_reph, ("x",), type(self).__name__)

        # Timing. Every interval between an RF centre and the echo it belongs
        # to is half a spacing, which is what CPMG means.
        exc_span = _span(system, rf, gz)
        ref_span = _span(system, rf_ref, gz_ref)
        rf_center = float(rf.delay) + float(rf.center)
        ref_center = float(rf_ref.delay) + float(rf_ref.center)
        lead_fixed = exc_span - rf_center + ref_center
        pre_fixed = ref_span - ref_center + echo_offset
        post_fixed = ref_center + flat_span - echo_offset

        encode_resolution = [fov[axis] / n[axis] for axis in range(1, ndim)]
        lead_floor = _span(system, gx_pre, gz_reph, trigger)
        pre_floor = _span(
            system, gx_bridge_pre, *_encodes(system, encode_resolution, 1.0, None)
        )
        post_floor = pp.ceil_to_raster(
            max(
                float(post_times[-1]),
                *(
                    pp.calc_duration(encode)
                    for encode in _encodes(system, encode_resolution, -1.0, None)
                    if encode is not None
                ),
            ),
            raster,
        )

        esp_min = pp.ceil_to_raster(
            2.0 * max(pre_fixed + pre_floor, post_fixed + post_floor), raster
        )
        if esp is None:
            esp = esp_min
        else:
            esp = pp.round_to_raster(float(esp), raster)
            if esp < esp_min - 1e-12:
                raise ValueError(
                    f"the requested echo spacing of {esp * 1e3:.3f} ms is shorter than the "
                    f"{esp_min * 1e3:.3f} ms this train can achieve"
                )

        # The first spacing is its own number, so a long excitation is paid for
        # once instead of by every echo. It differs from the rest by twice a
        # raster at a time, since half of it has to land on the raster too.
        lead_min = esp + 2.0 * pp.ceil_to_raster(
            max(0.0, lead_fixed + lead_floor - 0.5 * esp), raster
        )
        if esp_first is None:
            esp_first = lead_min
        else:
            esp_first = esp + 2.0 * pp.round_to_raster(
                0.5 * (float(esp_first) - esp), raster
            )
            if esp_first < lead_min - 1e-12:
                raise ValueError(
                    f"the requested first echo spacing of {esp_first * 1e3:.3f} ms is shorter "
                    f"than the {lead_min * 1e3:.3f} ms needed to reach the first refocusing "
                    f"pulse from this excitation; leave esp_first unset to be given that, or "
                    f"lengthen esp until half of it covers the excitation"
                )
        wait_first = 0.5 * (esp_first - esp)

        lead_span = max(
            lead_floor, pp.round_to_raster(0.5 * esp_first - lead_fixed, raster)
        )
        pre_span = pp.round_to_raster(0.5 * esp - pre_fixed, raster)
        pre_span = min(
            max(pre_span, pre_floor), esp - ref_span - flat_span - post_floor
        )
        post_span = esp - ref_span - flat_span - pre_span

        # Solved spans in hand, the encodes are rebuilt to fill their windows:
        # the same area over a longer time is a lower amplitude and a gentler
        # slew, and it costs nothing, the window being fixed by the spacing.
        gy_pre, gz_pre = _encodes(system, encode_resolution, 1.0, pre_span)
        gy_rew, gz_rew = _encodes(system, encode_resolution, -1.0, post_span)
        gx_pre.delay = lead_span - pp.calc_duration(gx_pre)
        gx_bridge_pre.delay = pre_span - pp.calc_duration(gx_bridge_pre)
        # Both read lobes span their whole block, the entry one by starting
        # late and the exit one by running on at zero. That is what keeps the
        # echo spacing a property of the read axis alone: a loop that leaves an
        # encode out -- a multi-echo spin echo does -- still gets this timing.
        if post_span > post_times[-1] + 1e-12:
            post_times = np.append(post_times, post_span)
            post_amplitudes = np.append(post_amplitudes, 0.0)
        gx_bridge_post = pp.make_extended_trapezoid(
            channel="x", times=post_times, amplitudes=post_amplitudes, system=system
        )

        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, *present(gz))
        self.seq.add_block(gx_pre, *present(gz_reph), *present(trigger))
        for echo in range(etl):
            self.seq.add_block(rf_ref, *present(gz_ref))
            if wait_first and echo == 0:
                wait_esp1 = pp.make_delay(wait_first)
                self.seq.add_block(wait_esp1)
            self.seq.add_block(gx_bridge_pre, gy_pre, *present(gz_pre))
            self.seq.add_block(
                gx, adc, *present(gy_wave), *present(gz_wave), *adc_labels
            )
            self.seq.add_block(gx_bridge_post, gy_rew, *present(gz_rew))

        tr_min = self.seq.duration()[0]
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            self.seq.add_block(wait_tr)

        first_echo = (
            exc_span + lead_span + ref_span + wait_first + pre_span + echo_offset
        )
        self.center = first_echo
        self.echo_times = first_echo - rf_center + esp * np.arange(etl)
        self.esp = esp
        self.esp_first = esp_first
        self.etl = etl
        self.bandwidth_hz = 1.0 / dwell
        self.n_samples = n_samples
        self.center_sample = n_pre
        self.delta_kx = delta_kx
        self.readout_duration = readout_duration
        # What the corkscrew reached, which is the requested amplitude only
        # where the slew rate left room for it.
        self.wave_amplitude = wave_peak


def _span(system: pp.Opts, *events: Any) -> float:
    """Where a block holding ``events`` ends, ignoring the ones that are None."""
    events = [event for event in events if event is not None]
    if not events:
        return 0.0
    return pp.ceil_to_raster(pp.calc_duration(*events), system.block_duration_raster)


def _encodes(
    system: pp.Opts, resolutions, sign: float, duration: float | None
) -> tuple:
    """One phase encode per encoded axis after the readout, y first.

    Always a pair, ``None`` standing in for the partition encode of a 2D train,
    so the caller can unpack it either way.
    """
    encodes = [
        pp.scale_grad(
            pp.make_phase_encoding(axis, resolution, system=system, duration=duration),
            sign,
        )
        for axis, resolution in zip("yz", resolutions, strict=False)
    ]
    encodes.append(None)
    return tuple(encodes[:2])


class FseReadout2D(_FseReadout):
    """A slice-selective CPMG train, frequency-encoded along x.

    ``fov`` and ``matrix`` take two values, readout first. See
    :class:`_FseReadout` for the timing, crushing and encoding arguments.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 90.0, 5e-3)
    >>> refocusing = design.SpatialSelectiveRefocusing(system, 5e-3)
    >>> fse = design.FseReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     rf_ref=refocusing.rf_ref, gz_ref=refocusing.gz,
    ...     fov=0.22, matrix=192, etl=8,
    ... )
    >>> fse.etl, int(fse.adc.num_samples)
    (8, 192)

    Every echo replays the same events:

    >>> fse.blocks[-3] == (fse.gx_bridge_pre, fse.gy_pre)
    True

    Which line each echo encodes is the loop's, and it is the whole of the
    contrast: this train is played in centric order, so the k-space centre is
    acquired at the first echo and the effective echo time is one spacing.

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import trajectory

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       excitation = design.SpatialSelectiveExcitation(system, 90.0, 3e-3)
       refocusing = design.SpatialSelectiveRefocusing(system, 3e-3)
       fse = design.FseReadout2D(
           system, excitation.rf, excitation.gz, excitation.gz_reph,
           rf_ref=refocusing.rf_ref, gz_ref=refocusing.gz,
           fov=0.22, matrix=128, etl=5, readout_bandwidth_hz=100e3,
       )
       trajectory(
           fse,
           ky=[0.0, 0.25, -0.25, 0.5, -0.5],
           title="FseReadout2D, five echoes in centric order",
       )

    One excitation and the train that follows it, echo spacing and all:

    .. plot::
       :include-source:

       import pulserver.design as design
       import pulserver.pypulseq as pp

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       excitation = design.SpatialSelectiveExcitation(system, 90.0, 3e-3)
       refocusing = design.SpatialSelectiveRefocusing(system, 3e-3)
       fse = design.FseReadout2D(
           system, excitation.rf, excitation.gz, excitation.gz_reph,
           rf_ref=refocusing.rf_ref, gz_ref=refocusing.gz,
           fov=0.22, matrix=128, etl=5, readout_bandwidth_hz=100e3,
       )
       fse.plot(time_disp="ms", grad_disp="mT/m", stacked=True, plot_now=False)
    """

    _ndim = 2


class FseReadout3D(_FseReadout):
    """A single-slab CPMG train, phase-encoded along y and partition-encoded along z.

    ``fov`` and ``matrix`` take three values, readout first. Refocus
    non-selectively and raise ``spoiling_cycles``: that pairing is what makes
    single-slab 3D FSE efficient.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> slab = design.SpatialSelectiveExcitation(system, 90.0, 0.16, is_slab=True)
    >>> refocusing = design.NonSelectiveRefocusing(
    ...     system, duration_s=0.6e-3, spoiling_cycles=0.0
    ... )
    >>> fse = design.FseReadout3D(
    ...     system, slab.rf, slab.gz, rf_ref=refocusing.rf_ref,
    ...     fov=(0.22, 0.22, 0.16), matrix=(192, 192, 80), etl=32,
    ...     spoiling_cycles=2.0,
    ... )

    Both encodes are trapezoids -- one waveform at many amplitudes:

    >>> fse.gy_pre.type, fse.gz_pre.type
    ('trap', 'trap')

    The slab pulse does not fit in half a spacing, so the first echo pays for
    it and the other thirty-one do not:

    >>> fse.esp_first > fse.esp
    True

    Two encodes per echo, so the train is a path through the ``(ky, kz)``
    plane rather than a stack of lines. Where that path starts and how fast
    it leaves the centre is the ordering the loop chose:

    .. plot::

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import trajectory

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       slab = design.SpatialSelectiveExcitation(system, 90.0, 0.16, is_slab=True)
       refocusing = design.NonSelectiveRefocusing(
           system, duration_s=0.6e-3, spoiling_cycles=0.0
       )
       fse = design.FseReadout3D(
           system, slab.rf, slab.gz, rf_ref=refocusing.rf_ref,
           fov=(0.22, 0.22, 0.16), matrix=(128, 128, 32), etl=5,
           spoiling_cycles=2.0, readout_bandwidth_hz=100e3,
       )
       trajectory(
           fse,
           ky=[0.0, 0.3, -0.3, 0.6, -0.6],
           kz=[0.0, 0.2, 0.4, 0.6, 0.8],
           plane="yz",
           title="FseReadout3D, five echoes in the encoding plane",
       )
    """

    _ndim = 3
