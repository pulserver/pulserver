"""Decoding a block's event ids back into PyPulseq events.

The sequence stores blocks as integer ids into its libraries; these turn a
row back into the ``SimpleNamespace`` a PyPulseq caller expects, and
:func:`block_to_events` flattens that namespace back into the argument list
:meth:`Sequence.add_block` takes.
"""

from __future__ import annotations

__all__ = ["block_to_events"]

from types import SimpleNamespace

import numpy as np

from ._events import _shape_dur


#: Pulseq's trigger numbering, as ``kind -> channel number -> name``.
_TRIGGER_CHANNELS = {
    1.0: {1.0: "osc0", 2.0: "osc1", 3.0: "ext1"},
    2.0: {1.0: "physio1", 2.0: "physio2"},
}


def _decompress(num_samples: int, samples: np.ndarray) -> np.ndarray:
    """One shape's samples, run-length decoded if it was compressed.

    Pulseq stores the quantised derivative and lets a run of three or more
    equal values collapse to two of them plus a repeat count. A shape whose
    stored length already equals its sample count was left uncompressed.
    """
    if samples.size == num_samples:
        return np.asarray(samples, dtype=float)

    derivative = np.empty(num_samples, dtype=float)
    written = 0
    position = 0
    while position < samples.size:
        value = samples[position]
        derivative[written] = value
        written += 1
        position += 1
        if position < samples.size and samples[position] == value:
            derivative[written] = value
            written += 1
            position += 1
            if position < samples.size:
                repeats = round(float(samples[position]))
                position += 1
                derivative[written : written + repeats] = value
                written += repeats
    return np.cumsum(derivative[:written])


def _shape(native, shape_id: int) -> np.ndarray:
    """Library shape ``shape_id``, decompressed."""
    num_samples, samples = native.shape_row(shape_id)
    return _decompress(num_samples, samples)


def _times(native, time_shape_id: int, count: int, raster: float) -> np.ndarray:
    """The sample times of a waveform, materialised on its raster.

    Zero means the standard raster, where sample ``i`` sits at the centre of
    its interval; ``-1`` is the half-raster variant a gradient may use;
    anything else is a shape of ticks.
    """
    if time_shape_id > 0:
        return _shape(native, time_shape_id) * raster
    if time_shape_id < 0:
        return 0.5 * raster * np.arange(1, count + 1)
    return (np.arange(count) + 0.5) * raster


def _rf_event(native, rf_id: int) -> SimpleNamespace:
    """RF library row ``rf_id`` as a PyPulseq RF event."""
    row = native.rf_row(rf_id)
    magnitude = _shape(native, int(row[1]))
    phase = _shape(native, int(row[2]))
    raster = native.rf_raster_time
    return SimpleNamespace(
        type="rf",
        signal=row[0] * magnitude * np.exp(2j * np.pi * phase),
        t=_times(native, int(row[3]), magnitude.size, raster),
        shape_dur=magnitude.size * raster,
        center=row[4],
        delay=row[5],
        freq_ppm=row[6],
        phase_ppm=row[7],
        freq_offset=row[8],
        phase_offset=row[9],
        dead_time=0.0,
        ringdown_time=0.0,
        use=native.rf_use(rf_id),
    )


def _grad_event(native, grad_id: int, channel: str) -> SimpleNamespace:
    """Gradient ``grad_id`` as a PyPulseq trapezoid or arbitrary gradient."""
    row = native.grad_row(grad_id)
    if native.grad_kind(grad_id) == "trap":
        amplitude, rise, flat, fall, delay = row
        return SimpleNamespace(
            type="trap",
            channel=channel,
            amplitude=amplitude,
            rise_time=rise,
            flat_time=flat,
            fall_time=fall,
            delay=delay,
            area=amplitude * (flat + rise / 2 + fall / 2),
            flat_area=amplitude * flat,
            first=0.0,
            last=0.0,
        )

    raster = native.grad_raster_time
    waveform = row[0] * _shape(native, int(row[3]))
    times = _times(native, int(row[4]), waveform.size, raster)
    return SimpleNamespace(
        type="grad",
        channel=channel,
        waveform=waveform,
        tt=times,
        # Vertex times or sample centres, told apart by whether the first is
        # zero -- an extended trapezoid stores four vertices spanning
        # milliseconds, so a sample count times the raster is not its length.
        shape_dur=_shape_dur(times),
        first=row[1],
        last=row[2],
        delay=row[5],
        area=float(np.sum(waveform) * raster),
    )


def _adc_event(native, adc_id: int) -> SimpleNamespace:
    """ADC library row ``adc_id`` as a PyPulseq ADC event."""
    row = native.adc_row(adc_id)
    return SimpleNamespace(
        type="adc",
        num_samples=round(float(row[0])),
        dwell=row[1],
        delay=row[2],
        freq_ppm=row[3],
        phase_ppm=row[4],
        freq_offset=row[5],
        phase_offset=row[6],
        dead_time=0.0,
        phase_modulation=_shape(native, int(row[7])) if row[7] else None,
    )


def _decode_extensions(native, head: int, block: SimpleNamespace) -> None:
    """Walk a block's extension chain, filling in what each node carries."""
    node = head
    while node:
        type_id, reference, node = native.extension_row(node)
        name = native.extension_type_name(type_id)
        if name == "TRIGGERS":
            row = native.trigger_row(reference)
            block.triggers.append(
                SimpleNamespace(
                    type="trigger" if row[0] == 2.0 else "output",
                    channel=_TRIGGER_CHANNELS.get(row[0], {}).get(row[1], "osc0"),
                    delay=row[2],
                    duration=row[3],
                )
            )
        elif name == "LABELSET":
            value, label_id = native.label_set_row(reference)
            block.label_sets.append((native.label_name(label_id), value))
            block.labels.append(("SET", native.label_name(label_id), value))
        elif name == "LABELINC":
            value, label_id = native.label_inc_row(reference)
            block.label_incs.append((native.label_name(label_id), value))
            block.labels.append(("INC", native.label_name(label_id), value))
        elif name == "ROTATIONS":
            block.rotation = native.rotation_row(reference)
        elif name == "RF_SHIMS":
            channels = native.rf_shim_row(reference).reshape(-1, 2)
            block.rf_shim = channels[:, 0] * np.exp(1j * channels[:, 1])
        elif name == "DELAYS":
            num, offset, factor, hint = native.soft_delay_row(reference)
            block.soft_delay = SimpleNamespace(
                type="soft_delay", numID=num, offset=offset, factor=factor, hint=hint
            )


def block_to_events(block: SimpleNamespace) -> tuple:
    """A decoded block flattened back into the arguments ``add_block`` takes.

    The inverse of :meth:`~pulserver.pypulseq.Sequence.get_block`, so a
    sequence can be rebuilt block by block -- which is how a module edits one
    of its own blocks after the fact, adding a label to a block a transform has
    already annotated::

        rebuilt = pp.Sequence(system)
        for index in range(1, built.num_blocks + 1):
            events = pp.block_to_events(built.get_block(index))
            rebuilt.add_block(*events, *(labels if index == 1 else ()))

    The extensions come back as events rather than as the raw rows
    :meth:`get_block` reports them in: labels through
    :func:`~pypulseq.make_label`, the rotation through
    :func:`~pulserver.pypulseq.make_rotation`, the shim through
    :func:`~pulserver.pypulseq.make_rf_shim`. A block held open longer than its
    events reach also gets a delay, so its duration survives the round trip.

    Parameters
    ----------
    block : types.SimpleNamespace
        A block as :meth:`get_block` returns it.

    Returns
    -------
    tuple
        The block's events, in the order Pulseq plays them.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> seq = pp.Sequence(system)
    >>> _ = seq.add_block(pp.make_delay(1e-3), pp.make_label(type="SET", label="LIN", value=3))
    >>> events = pp.block_to_events(seq.get_block(1))
    >>> sorted(event.type for event in events)
    ['delay', 'labelset']
    """
    import pypulseq

    from ._make_label import make_label
    from ._make_rf_shim import make_rf_shim
    from ._make_rotation import make_rotation

    events = [
        event
        for event in (block.rf, block.gx, block.gy, block.gz, block.adc)
        if event is not None
    ]
    events.extend(block.triggers)
    events.extend(
        make_label(type=kind, label=name, value=value) for kind, name, value in block.labels
    )
    if block.rotation is not None:
        events.append(make_rotation(_as_rotation(block.rotation)))
    if block.rf_shim is not None:
        events.append(make_rf_shim(block.rf_shim))
    if block.soft_delay is not None:
        delay = block.soft_delay
        events.append(
            pypulseq.make_soft_delay(
                hint=delay.hint, numID=delay.numID, offset=delay.offset, factor=delay.factor
            )
        )

    if _span(events) < block.block_duration - 1e-12:
        events.append(pypulseq.make_delay(block.block_duration))
    return tuple(events)


def _span(events: list) -> float:
    """How far the events themselves reach (s)."""
    import pypulseq

    timed = [event for event in events if getattr(event, "type", None) in _TIMED]
    return pypulseq.calc_duration(*timed) if timed else 0.0


#: Event kinds that occupy time. The rest are annotations on the block.
_TIMED = frozenset({"rf", "trap", "grad", "adc", "delay", "trigger", "output"})


def _as_rotation(quaternion):
    """A stored ``(w, x, y, z)`` row as the rotation :func:`make_rotation` wants."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(np.asarray(quaternion, dtype=float), scalar_first=True)
