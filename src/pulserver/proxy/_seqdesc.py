"""The event stream of a sequence, as a simulation reads it.

The stream records when each pulse turns the magnetization, through what
angle and about what phase, and when each readout samples it. It carries no
gradients and no waveform memory.

It is read mechanically off the Pulseq file. Which signal model it means is
not: a Pulseq file has no gradient use field, so a crusher cannot be told from
a phase encode without tracking the k-space moment through a repetition, and
whether a readout is followed by spoiling is a property of the sequence's
intent. That choice belongs to the reconstruction plugin, against the sequence
family in hand.

The field order of each event is the order the simulation reads it back in.
"""

from __future__ import annotations

__all__ = [
    "AdcRole",
    "EventType",
    "RfUse",
    "SequenceDescription",
    "SequenceEvent",
    "describe",
    "is_description",
    "waveforms",
]

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import ismrmrd
import numpy as np
import pypulseqpp as pp

#: Band offset slots every RF definition carries, filled or not.
BANDS = 8

#: Waveform ids of a description: one header and one event list per
#: subsequence, one waveform per RF definition. ISMRMRD reserves ids below
#: 1024 for its own physiological channels.
HEADER_ID = 1024
EVENTS_ID = 1025
RF_DEFINITION_ID = 1026
#: Layout of the words those waveforms carry.
FORMAT_VERSION = 1
#: Samples one waveform channel holds.
_CHANNEL_SAMPLES = 65535
#: Words an event takes in the event list: type, timestamp (two), parameters.
EVENT_WORDS = 10


class EventType(IntEnum):
    """What one row of the stream is."""

    WAIT = 0
    RF = 1
    ADC = 2


class RfUse(IntEnum):
    """Pulseq's RF-use tag, all seven of them."""

    UNKNOWN = 0
    EXCITATION = 1
    REFOCUSING = 2
    INVERSION = 3
    SATURATION = 4
    PREPARATION = 5
    OTHER = 6


class AdcRole(IntEnum):
    """How one readout stands among the readouts of its repetition."""

    NON_ACQUIRED = 0
    SINGLE = 1
    ECHO_CENTER = 2
    NON_CENTER = 3


_USE_TAGS = {
    "excitation": RfUse.EXCITATION,
    "refocusing": RfUse.REFOCUSING,
    "inversion": RfUse.INVERSION,
    "saturation": RfUse.SATURATION,
    "preparation": RfUse.PREPARATION,
    "other": RfUse.OTHER,
}


@dataclass(frozen=True)
class RfShape:
    """One sampled shape of a pulse."""

    num_uncompressed: int
    samples: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class RfDefinition:
    """A pulse the stream's RF events name.

    ``magnitude`` is normalised to a peak of one and scaled by the amplitude of
    the event that plays it, so one definition serves every instance of a pulse
    whatever angle it turns through. ``phase`` is in turns, not radians, and
    ``time`` holds each sample's centre in units of the RF raster.
    """

    id: int
    bandwidth_hz: float
    num_bands: int
    band_frequency_offsets_hz: tuple[float, ...]
    band_bandwidth_hz: float
    magnitude: RfShape
    phase: RfShape | None = None
    time: RfShape | None = None


@dataclass(frozen=True)
class SequenceEvent:
    """One WAIT, RF or ADC row, at the time it happens.

    An RF row carries ``(definition_id, use, amplitude_hz, phase_rad,
    frequency_hz, shim_id, slice_select_gradient_hz_per_m)``; an ADC row
    ``(role, phase_rad, is_echo)``; a WAIT row nothing. An RF row's phase is
    the one at the pulse's centre, its frequency offset accrued from the
    pulse's start; phases and frequencies hold the ppm offsets resolved at the
    design's field.
    """

    type: EventType
    timestamp_us: float
    params: tuple[Any, ...] = ()


@dataclass(frozen=True)
class SequenceDescription:
    """The stream of one repetition of a subsequence and the pulses it names.

    Times run from the start of the repetition, whose length is
    ``tr_duration_us``.
    """

    subsequence_index: int
    tr_duration_us: float
    events: tuple[SequenceEvent, ...]
    rf_definitions: dict[int, RfDefinition]
    rf_raster_time_s: float = 1e-6

    def __len__(self) -> int:
        return len(self.events)


def describe(seq: Any, *, subsequence_index: int = 0) -> SequenceDescription:
    """Read the event stream of one repetition of a sequence.

    The repetition is the one whose readout passes nearest the centre of
    k-space, navigators aside: in a dual-region echo train, the train that
    samples the centre. Repetitions are ``pypulseqpp.Sequence.repetition``'s,
    so a sequence that does not repeat, such as a fingerprinting train, is
    described whole. One row per block of it, in play order: the pulse a block
    turns the magnetization through, the readout it samples with, or a wait
    where it does neither.

    An RF row is timed at the pulse's centre, and a readout at the sample
    passing nearest the centre of k-space -- the time a simulation has to place
    an echo at, which is not the middle of the acquisition window when the
    readout is asymmetric.

    Parameters
    ----------
    seq : pypulseqpp.Sequence
        A ``pypulseqpp.Sequence``.
    subsequence_index : int, default=0
        Which subsequence of a chain this is.

    Returns
    -------
    SequenceDescription
    """
    held = seq.block_durations
    durations = np.asarray(
        [float(held[block]) for block in range(1, seq.num_blocks + 1)]
    )
    starts = np.concatenate(([0.0], np.cumsum(durations)))
    pulses = seq.rf_times(compat=False)
    echoes = seq.adc_echoes()
    first_block, last_block = _central_repetition(seq, echoes)
    origin = starts[first_block - 1]
    within = range(first_block, last_block + 1)
    gradients = seq.rf_gradients()
    slice_select = _slice_select(gradients)

    # Read once for the whole sequence: each of these tabulates every block,
    # and reading it again per pulse makes a long scan quadratic.
    library_ids = _rf_library_ids(seq)
    shims = _shim_ids(seq)

    definitions: dict[int, RfDefinition] = {}
    rows: dict[int, SequenceEvent] = {}

    for at, block in enumerate(np.asarray(pulses.block, dtype=int)):
        if not first_block <= block <= last_block:
            continue
        pulse = seq.get_block(int(block)).rf
        definition = int(library_ids[block - 1])
        if definition not in definitions:
            definitions[definition] = _definition_of(
                definition, pulse, float(seq.rf_raster_time)
            )
        rows[int(block)] = SequenceEvent(
            EventType.RF,
            (float(pulses.t[at]) - origin) * 1e6,
            (
                definition,
                int(_USE_TAGS.get(str(pulses.use[at]).lower(), RfUse.UNKNOWN)),
                float(np.max(np.abs(np.asarray(pulse.signal)))),
                float(pulses.phase_offset[at]),
                float(pulses.freq_offset[at]),
                int(shims[block - 1]),
                slice_select.get(int(block), 0.0),
            ),
        )

    samples = np.asarray(seq.adc_times()[0])
    for at, block in enumerate(np.asarray(echoes.block, dtype=int)):
        if not first_block <= block <= last_block:
            continue
        adc = seq.get_block(int(block)).adc
        # A readout with no echo, one played without a gradient, is at the
        # centre from its first sample.
        centre = max(int(np.asarray(echoes.echo)[at].min()), 0)
        first = int(np.asarray(echoes.first_sample)[at])
        rows[int(block)] = SequenceEvent(
            EventType.ADC,
            (float(samples[first + centre]) - origin) * 1e6,
            (
                int(AdcRole.SINGLE),
                pp.calc_absolute_offsets(adc, system=seq.system)[1],
                1,
            ),
        )

    events = tuple(
        rows.get(
            block,
            SequenceEvent(EventType.WAIT, (float(starts[block - 1]) - origin) * 1e6),
        )
        for block in within
    )
    return SequenceDescription(
        subsequence_index=subsequence_index,
        tr_duration_us=float(starts[last_block] - origin) * 1e6,
        events=events,
        rf_definitions=definitions,
        rf_raster_time_s=float(seq.rf_raster_time),
    )


def _central_repetition(seq: Any, echoes: Any) -> tuple[int, int]:
    """Return the first and last block of the repetition sampling the centre of k-space.

    The readout nearest the centre is the one whose sample at its echo is,
    among readouts not labelled ``NAV``; the whole sequence when it has no
    such readout or does not repeat.
    """
    size, _ = seq.repetition()
    blocks = np.asarray(echoes.block, dtype=int)
    if size <= 0 or size >= seq.num_blocks or not blocks.size:
        return 1, seq.num_blocks
    k = np.asarray(seq.calculate_kspace()[0])
    centres = np.asarray(echoes.first_sample, dtype=int) + np.asarray(
        [max(int(np.asarray(echo).min()), 0) for echo in echoes.echo]
    )
    radius = np.linalg.norm(k[:, centres], axis=0)
    navigator = np.asarray(
        seq.evaluate_labels(evolution="adc").get("NAV", np.zeros(blocks.size)),
        dtype=bool,
    ).reshape(-1)
    candidates = np.flatnonzero(~navigator) if navigator.size == blocks.size else []
    if not len(candidates):
        candidates = np.arange(blocks.size)
    # The first repetition reaching the centre, within rounding of k.
    reach = radius[candidates]
    nearest = int(
        candidates[
            np.flatnonzero(reach <= reach.min() + 1e-6 * (reach.max() or 1.0))[0]
        ]
    )
    first = (int(blocks[nearest]) - 1) // size * size + 1
    return first, min(first + size - 1, seq.num_blocks)


def _slice_select(gradients: Any) -> dict[int, float]:
    """Return the gradient each pulse plays under, as one amplitude per block."""
    blocks = np.asarray(gradients.block, dtype=int)
    played = np.asarray(gradients.gradient, dtype=float).reshape(len(blocks), 3)
    return {
        int(block): float(played[at][np.argmax(np.abs(played[at]))])
        for at, block in enumerate(blocks)
    }


def _rf_library_ids(seq: Any) -> np.ndarray:
    """Return the library id of the pulse each block plays, block ``i`` at ``i - 1``."""
    return np.asarray([row[1] for row in seq.block_events.values()], dtype=int)


def _shim_ids(seq: Any) -> np.ndarray:
    """Return the transmit shim each block drives, block ``i`` at ``i - 1``; 0 where none."""
    return np.asarray(seq.block_shims(), dtype=int)


def _definition_of(definition: int, pulse: Any, raster_s: float) -> RfDefinition:
    """Read a pulse into a definition, its magnitude normalised to a peak of one."""
    signal = np.asarray(pulse.signal)
    peak = float(np.max(np.abs(signal))) or 1.0
    magnitude = RfShape(signal.size, np.abs(signal) / peak)
    # Turns, not radians: what reads this applies its own factor of 2 pi.
    phase = RfShape(signal.size, np.angle(signal) / (2.0 * np.pi))
    time = RfShape(signal.size, np.asarray(pulse.t) / raster_s)
    try:
        bandwidth = float(pp.calc_rf_bandwidth(pulse, compat=False).bandwidth)
    except (ValueError, ZeroDivisionError):
        bandwidth = 0.0
    return RfDefinition(
        id=definition,
        bandwidth_hz=bandwidth if np.isfinite(bandwidth) else 0.0,
        num_bands=1,
        band_frequency_offsets_hz=(0.0,) * BANDS,
        band_bandwidth_hz=0.0,
        magnitude=magnitude,
        phase=phase,
        time=time,
    )


def as_rows(description: SequenceDescription) -> dict[str, np.ndarray]:
    """Return the stream as the flat arrays a wire format carries.

    ``type`` and ``timestamp_us`` hold one value per event and ``params`` an
    ``(n, 7)`` row each, short rows padded with zeros: one row per block over a
    whole pass, never a record per event.
    """
    count = len(description.events)
    kinds = np.empty(count, dtype=np.int32)
    # Microseconds from the start of the sequence outgrow float32's 24-bit
    # mantissa within seconds.
    times = np.empty(count, dtype=np.float64)
    params = np.zeros((count, 7), dtype=np.float32)
    for at, event in enumerate(description.events):
        kinds[at] = int(event.type)
        times[at] = event.timestamp_us
        for column, value in enumerate(event.params):
            params[at, column] = float(value)
    return {"type": kinds, "timestamp_us": times, "params": params}


def waveforms(
    descriptions: Sequence[SequenceDescription], measurement_uid: int = 0
) -> list[Any]:
    """Return the ISMRMRD waveforms carrying the descriptions of a chain.

    Per subsequence a header (:data:`HEADER_ID`), the event list
    (:data:`EVENTS_ID`) and one waveform per RF definition
    (:data:`RF_DEFINITION_ID`). Each holds one vector of 32-bit words, its
    length first, laid over as many channels as keep a channel within the
    65 535 samples a waveform allows, zero-padded; floats are bit-cast,
    ``float64`` values take two words, low then high.

    - Header: :data:`FORMAT_VERSION`, subsequence index, number of
      subsequences, number of events, number of RF definitions,
      ``tr_duration_us`` (float64), ``rf_raster_time_s`` (float64).
    - Event list: subsequence index, number of events, then
      :data:`EVENT_WORDS` words an event: type, ``timestamp_us`` (float64),
      the seven parameters (float32) of :class:`SequenceEvent`.
    - RF definition: subsequence index, id, ``bandwidth_hz``, ``num_bands``,
      the :data:`BANDS` band offsets, ``band_bandwidth_hz``,
      ``total_b1sq_power`` (0), then for the magnitude, phase and time shapes
      in turn whether it is given, its uncompressed and its stored sample
      counts, and then the samples (float32) of the shapes given, in that
      order.
    """
    found = []
    for description in descriptions:
        index = description.subsequence_index
        found.append(
            _waveform(
                HEADER_ID,
                [
                    _u32([FORMAT_VERSION, index, len(descriptions)]),
                    _u32([len(description.events), len(description.rf_definitions)]),
                    _f64([description.tr_duration_us, description.rf_raster_time_s]),
                ],
                measurement_uid,
            )
        )
        rows = as_rows(description)
        count = rows["type"].size
        events = np.empty((count, EVENT_WORDS), dtype=np.uint32)
        events[:, 0] = rows["type"].astype(np.uint32)
        events[:, 1:3] = (
            rows["timestamp_us"].astype("<f8").view("<u4").reshape(count, 2)
        )
        events[:, 3:] = rows["params"].astype("<f4").view("<u4")
        found.append(
            _waveform(
                EVENTS_ID,
                [_u32([index, count]), events.reshape(-1)],
                measurement_uid,
            )
        )
        for definition in description.rf_definitions.values():
            shapes = (definition.magnitude, definition.phase, definition.time)
            words = [
                _u32([index, definition.id]),
                _f32([definition.bandwidth_hz]),
                _u32([definition.num_bands]),
                _f32(list(definition.band_frequency_offsets_hz)),
                _f32([definition.band_bandwidth_hz, 0.0]),
            ]
            for shape in shapes:
                words.append(
                    _u32([0, 0, 0])
                    if shape is None
                    else _u32([1, shape.num_uncompressed, len(shape.samples)])
                )
            words += [_f32(shape.samples) for shape in shapes if shape is not None]
            found.append(_waveform(RF_DEFINITION_ID, words, measurement_uid))
    return found


def is_description(item: Any) -> bool:
    """Whether a stream item is one of the waveforms :func:`waveforms` writes."""
    return isinstance(item, ismrmrd.Waveform) and int(item.waveform_id) in (
        HEADER_ID,
        EVENTS_ID,
        RF_DEFINITION_ID,
    )


def _u32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="<u4")


def _f32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="<f4").view("<u4")


def _f64(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="<f8").view("<u4")


def _waveform(identifier: int, words: list[np.ndarray], measurement_uid: int) -> Any:
    """Lay ``words``, their count first, over the channels of one waveform."""
    payload = np.concatenate([np.asarray(w, dtype="<u4").reshape(-1) for w in words])
    payload = np.concatenate((_u32([payload.size]), payload))
    channels = max(1, -(-payload.size // _CHANNEL_SAMPLES))
    samples = -(-payload.size // channels)
    data = np.zeros(channels * samples, dtype=np.uint32)
    data[: payload.size] = payload
    wave = ismrmrd.Waveform.from_array(data.reshape(channels, samples))
    wave.waveform_id = identifier
    wave.measurement_uid = measurement_uid
    wave.sample_time_us = 1.0
    return wave
