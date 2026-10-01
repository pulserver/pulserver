"""The event stream of a sequence, as a simulation reads it.

A reconstruction that simulates the signal needs what the sequence did to the
magnetisation: when each pulse turned it, through what angle and about what
phase, and when each readout sampled it. It does not need gradients, which it
would discard, nor the waveform memory the scanner lays out.

What is written here is mechanical -- it is read off the file the scanner
plays. Which signal model the stream means is not: a Pulseq file has no
gradient "use" field, so a crusher cannot be told from a phase encode without
tracking the k-space moment through a whole repetition, and whether a readout
is followed by spoiling is a property of the sequence's intent. That reading is
hand-written on the reconstruction side, against the sequence family in hand,
and nothing here guesses at it.

The field order of each event is the one the simulation reads them back in.
"""

from __future__ import annotations

__all__ = [
    "AdcRole",
    "EventType",
    "RfUse",
    "SequenceDescription",
    "SequenceEvent",
    "describe",
    "is_message",
    "message",
]

import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

#: Band offset slots every RF definition carries, filled or not.
BANDS = 8

#: The key an MRD text message carrying descriptions opens with.
MESSAGE_KEY = "pulserver_sequence_description"


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
    ``(role, phase_rad, is_echo)``; a WAIT row nothing.
    """

    type: EventType
    timestamp_us: float
    params: tuple[Any, ...] = ()


@dataclass(frozen=True)
class SequenceDescription:
    """The stream of one subsequence and the pulses it names."""

    subsequence_index: int
    tr_duration_us: float
    events: tuple[SequenceEvent, ...]
    rf_definitions: dict[int, RfDefinition]
    rf_raster_time_s: float = 1e-6

    def __len__(self) -> int:
        return len(self.events)


def describe(seq: Any, *, subsequence_index: int = 0) -> SequenceDescription:
    """Read the event stream of a sequence.

    One row per block, in play order: the pulse a block turns the magnetisation
    through, the readout it samples with, or a wait where it does neither.

    An RF row is timed at the pulse's centre, and a readout at the sample
    passing nearest the centre of k-space -- the time a simulation has to place
    an echo at, which is not the middle of the acquisition window when the
    readout is asymmetric.

    Parameters
    ----------
    seq
        A ``pypulseqpp.Sequence``.
    subsequence_index
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
    gradients = seq.rf_gradients()
    slice_select = _slice_select(gradients)

    definitions: dict[int, RfDefinition] = {}
    rows: dict[int, SequenceEvent] = {}

    for at, block in enumerate(np.asarray(pulses.block, dtype=int)):
        pulse = seq.get_block(int(block)).rf
        definition = _definition_id(seq, int(block))
        if definition not in definitions:
            definitions[definition] = _definition_of(
                definition, pulse, float(seq.rf_raster_time)
            )
        rows[int(block)] = SequenceEvent(
            EventType.RF,
            float(pulses.t[at]) * 1e6,
            (
                definition,
                int(_USE_TAGS.get(str(pulses.use[at]).lower(), RfUse.UNKNOWN)),
                float(np.max(np.abs(np.asarray(pulse.signal)))),
                float(pulses.phase_offset[at]),
                float(pulses.freq_offset[at]),
                _shim_id(seq, int(block)),
                slice_select.get(int(block), 0.0),
            ),
        )

    samples = np.asarray(seq.adc_times()[0])
    for at, block in enumerate(np.asarray(echoes.block, dtype=int)):
        adc = seq.get_block(int(block)).adc
        centre = int(np.asarray(echoes.echo)[at].min())
        first = int(np.asarray(echoes.first_sample)[at])
        rows[int(block)] = SequenceEvent(
            EventType.ADC,
            float(samples[first + centre]) * 1e6,
            (int(AdcRole.SINGLE), float(adc.phase_offset), 1),
        )

    events = tuple(
        rows.get(block, SequenceEvent(EventType.WAIT, float(starts[block - 1]) * 1e6))
        for block in range(1, seq.num_blocks + 1)
    )
    return SequenceDescription(
        subsequence_index=subsequence_index,
        tr_duration_us=float(np.sum(durations)) * 1e6,
        events=events,
        rf_definitions=definitions,
        rf_raster_time_s=float(seq.rf_raster_time),
    )


def _slice_select(gradients: Any) -> dict[int, float]:
    """Return the gradient each pulse plays under, as one amplitude per block."""
    blocks = np.asarray(gradients.block, dtype=int)
    played = np.asarray(gradients.gradient, dtype=float).reshape(len(blocks), 3)
    return {
        int(block): float(played[at][np.argmax(np.abs(played[at]))])
        for at, block in enumerate(blocks)
    }


def _definition_id(seq: Any, block: int) -> int:
    """Return the library id of the pulse a block plays."""
    return int(seq.block_events[block][1])


def _shim_id(seq: Any, block: int) -> int:
    """Return the transmit shim a block drives, 0 where it drives none."""
    shims = getattr(seq, "block_shims", None)
    if shims is None:
        return 0
    held = shims()
    return int(held.get(block, 0)) if isinstance(held, dict) else 0


def _definition_of(definition: int, pulse: Any, raster_s: float) -> RfDefinition:
    """Read a pulse into a definition, its magnitude normalised to a peak of one."""
    signal = np.asarray(pulse.signal)
    peak = float(np.max(np.abs(signal))) or 1.0
    magnitude = RfShape(signal.size, np.abs(signal) / peak)
    # Turns, not radians: what reads this applies its own factor of 2 pi.
    phase = RfShape(signal.size, np.angle(signal) / (2.0 * np.pi))
    time = RfShape(signal.size, np.asarray(pulse.t) / raster_s)
    return RfDefinition(
        id=definition,
        bandwidth_hz=0.0,
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


def message(descriptions: Sequence[SequenceDescription]) -> str:
    """Return the MRD text message carrying the descriptions of a chain.

    A JSON object holding one entry under :data:`MESSAGE_KEY`: a list with one
    object per subsequence, carrying ``subsequence_index``,
    ``tr_duration_us``, ``rf_raster_time_s``, the rows of :func:`as_rows` and
    ``rf_definitions``. Arrays are base64 of their little-endian bytes:
    ``type`` int32, ``timestamp_us`` float64, ``params`` float32 ``(n, 7)``
    row-major, and each shape's ``samples`` float32 beside its
    ``num_uncompressed``. A definition's ``phase`` and ``time`` may be
    ``null``.
    """
    return json.dumps({MESSAGE_KEY: [_subsequence(each) for each in descriptions]})


def is_message(item: Any) -> bool:
    """Whether a stream item is the text message :func:`message` writes."""
    return isinstance(item, str) and item.startswith('{"' + MESSAGE_KEY + '"')


def _subsequence(description: SequenceDescription) -> dict[str, Any]:
    rows = as_rows(description)
    return {
        "subsequence_index": description.subsequence_index,
        "tr_duration_us": description.tr_duration_us,
        "rf_raster_time_s": description.rf_raster_time_s,
        "type": _packed(rows["type"], "<i4"),
        "timestamp_us": _packed(rows["timestamp_us"], "<f8"),
        "params": _packed(rows["params"], "<f4"),
        "rf_definitions": [
            {
                "id": definition.id,
                "bandwidth_hz": definition.bandwidth_hz,
                "num_bands": definition.num_bands,
                "band_frequency_offsets_hz": list(definition.band_frequency_offsets_hz),
                "band_bandwidth_hz": definition.band_bandwidth_hz,
                "magnitude": _shape(definition.magnitude),
                "phase": _shape(definition.phase),
                "time": _shape(definition.time),
            }
            for definition in description.rf_definitions.values()
        ],
    }


def _shape(shape: RfShape | None) -> dict[str, Any] | None:
    if shape is None:
        return None
    return {
        "num_uncompressed": shape.num_uncompressed,
        "samples": _packed(shape.samples, "<f4"),
    }


def _packed(values: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode(
        "ascii"
    )
