"""Decode LiveSDK SEQDESC waveforms into a Python event model.

The scanner transports the sequence description as custom single-channel
ISMRMRD waveforms.  This module only decodes that transport; sequence-specific
physics is exposed through :mod:`pulserver.recon.simulation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable

import numpy as np

__all__ = [
    "AdcRole",
    "EventType",
    "RfDefinition",
    "RfShape",
    "RfUse",
    "SequenceDescription",
    "SequenceDescriptionCollection",
    "SequenceEvent",
    "SequenceParameters",
    "ShimDefinition",
    "decode_sequence_description",
    "decompress_shape",
]

WAVEFORM_ID_SEQDESC_HEADER = 999
WAVEFORM_ID_SEQDESC_EVENTS = 1000
WAVEFORM_ID_SEQDESC_RF_SHAPES = 1002
WAVEFORM_ID_SEQDESC_SHIMS = 1005
_SEQDESC_IDS = {
    WAVEFORM_ID_SEQDESC_HEADER,
    WAVEFORM_ID_SEQDESC_EVENTS,
    WAVEFORM_ID_SEQDESC_RF_SHAPES,
    WAVEFORM_ID_SEQDESC_SHIMS,
}


class EventType(IntEnum):
    """SEQDESC event type."""

    WAIT = 0
    RF = 1
    ADC = 2


class RfUse(IntEnum):
    """Pulseq RF-use tag."""

    UNKNOWN = 0
    EXCITATION = 1
    REFOCUSING = 2
    INVERSION = 3
    SATURATION = 4


class AdcRole(IntEnum):
    """Legacy relationship between ADCs in one canonical event stream."""

    NON_ACQUIRED = 0
    SINGLE = 1
    ECHO_CENTER = 2
    NON_CENTER = 3


@dataclass(frozen=True)
class SequenceParameters:
    """Scan-global SEQDESC header (waveform 999)."""

    num_subsequences: int
    min_te_us: float
    min_tr_us: float
    max_tr_us: float
    max_flip_angle_deg: float
    total_scan_time_us: float


@dataclass(frozen=True)
class SequenceEvent:
    """One WAIT, RF, or ADC state-machine event."""

    type: EventType
    timestamp_us: float
    params: tuple[float, ...]

    @property
    def rf_definition_id(self) -> int:
        self._require(EventType.RF)
        return int(self.params[0])

    @property
    def rf_use(self) -> RfUse:
        self._require(EventType.RF)
        return RfUse(int(self.params[1]))

    @property
    def rf_amplitude_hz(self) -> float:
        self._require(EventType.RF)
        return self.params[2]

    @property
    def rf_phase_rad(self) -> float:
        self._require(EventType.RF)
        return self.params[3]

    @property
    def rf_frequency_hz(self) -> float:
        self._require(EventType.RF)
        return self.params[4]

    @property
    def rf_shim_id(self) -> int:
        self._require(EventType.RF)
        return int(self.params[5])

    @property
    def slice_select_gradient_hz_per_m(self) -> float:
        self._require(EventType.RF)
        return self.params[6]

    @property
    def adc_role(self) -> AdcRole:
        self._require(EventType.ADC)
        return AdcRole(int(self.params[0]))

    @property
    def adc_phase_rad(self) -> float:
        self._require(EventType.ADC)
        return self.params[1]

    @property
    def is_echo(self) -> bool:
        """Whether any real instance of this ADC position reaches k~=0."""

        self._require(EventType.ADC)
        return bool(int(self.params[2]))

    def _require(self, expected: EventType) -> None:
        if self.type is not expected:
            raise AttributeError(
                f"{expected.name} payload requested from {self.type.name} event"
            )


@dataclass(frozen=True)
class RfShape:
    """One still-compressed Pulseq shape."""

    num_uncompressed: int
    samples: np.ndarray = field(repr=False, compare=False)

    def decompress(self, scale: float = 1.0) -> np.ndarray:
        """Return the Pulseq delta-RLE decoded shape."""

        return decompress_shape(self.samples, self.num_uncompressed, scale=scale)


@dataclass(frozen=True)
class RfDefinition:
    """RF definition referenced by RF events."""

    id: int
    bandwidth_hz: float
    num_bands: int
    band_frequency_offsets_hz: tuple[float, ...]
    band_bandwidth_hz: float
    total_b1sq_power: float
    magnitude: RfShape
    phase: RfShape | None = None
    time: RfShape | None = None

    def complex_envelope(self) -> np.ndarray:
        """Return the normalized complex RF envelope."""

        magnitude = self.magnitude.decompress()
        if self.phase is None:
            return magnitude.astype(np.complex64)
        phase = self.phase.decompress(scale=2.0 * np.pi)
        if phase.size != magnitude.size:
            raise ValueError(f"RF definition {self.id}: magnitude/phase size mismatch")
        return magnitude * np.exp(1j * phase)

    def flip_angle(
        self,
        amplitude_hz: float,
        *,
        rf_raster_time_s: float = 1e-6,
    ) -> tuple[float, float]:
        """Return ``(flip_rad, integral_phase_rad)`` for an RF occurrence.

        Pulseq stores RF amplitude in Hz, so the flip is
        ``2*pi*amplitude*abs(integral(envelope, time))``.
        """

        envelope = self.complex_envelope()
        if envelope.size < 2:
            return 0.0, 0.0
        if self.time is None:
            time_s = (
                np.arange(envelope.size, dtype=np.float64) + 0.5
            ) * rf_raster_time_s
        else:
            time_s = self.time.decompress(scale=rf_raster_time_s).astype(np.float64)
            if time_s.size != envelope.size:
                raise ValueError(
                    f"RF definition {self.id}: envelope/time size mismatch"
                )
        area = np.trapezoid(envelope, time_s)
        signed_area = float(amplitude_hz) * area
        return float(2.0 * np.pi * abs(signed_area)), float(np.angle(signed_area))


@dataclass(frozen=True)
class ShimDefinition:
    """One transmit-shim definition."""

    id: int
    magnitudes: tuple[float, ...]
    phases_rad: tuple[float, ...]


@dataclass(frozen=True)
class SequenceDescription:
    """Decoded event stream and RF resources for one subsequence."""

    subsequence_index: int
    tr_duration_us: float
    events: tuple[SequenceEvent, ...]
    rf_definitions: dict[int, RfDefinition]
    shim_definitions: dict[int, ShimDefinition]

    @property
    def adc_events(self) -> tuple[SequenceEvent, ...]:
        """ADC events in state-machine order."""

        return tuple(event for event in self.events if event.type is EventType.ADC)


@dataclass(frozen=True)
class SequenceDescriptionCollection:
    """Complete decoded SEQDESC resource set for a scan."""

    parameters: SequenceParameters
    subsequences: dict[int, SequenceDescription]

    def subsequence(self, index: int = 0) -> SequenceDescription:
        """Return one subsequence or raise a descriptive error."""

        try:
            return self.subsequences[index]
        except KeyError as exc:
            raise KeyError(f"SEQDESC has no subsequence {index}") from exc


def decompress_shape(
    packed: np.ndarray | Iterable[float],
    num_uncompressed: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Decode Pulseq derivative/RLE shape compression."""

    packed = np.asarray(packed, dtype=np.float32).reshape(-1)
    if num_uncompressed < 0:
        raise ValueError("negative uncompressed shape length")
    if packed.size == num_uncompressed:
        return (packed * scale).astype(np.float32, copy=True)
    if num_uncompressed == 0 and packed.size == 0:
        return np.empty(0, dtype=np.float32)

    delta = np.empty(num_uncompressed, dtype=np.float32)
    src = 0
    dst = 0
    while src < packed.size:
        value = packed[src]
        if src + 1 < packed.size and packed[src + 1] == value:
            if src + 2 >= packed.size:
                raise ValueError("truncated Pulseq RLE triplet")
            count_value = float(packed[src + 2]) + 2.0
            count = round(count_value)
            if abs(count_value - count) > 1e-6 or count < 2:
                raise ValueError("invalid Pulseq RLE repeat count")
            if dst + count > num_uncompressed:
                raise ValueError("Pulseq RLE expands beyond declared shape length")
            delta[dst : dst + count] = value
            dst += count
            src += 3
        else:
            if dst >= num_uncompressed:
                raise ValueError("Pulseq shape expands beyond declared length")
            delta[dst] = value
            dst += 1
            src += 1
    if dst != num_uncompressed:
        raise ValueError(f"Pulseq shape expanded to {dst}, expected {num_uncompressed}")
    return (np.cumsum(delta, dtype=np.float32) * scale).astype(np.float32)


def decode_sequence_description(
    waveforms: Iterable[Any],
) -> SequenceDescriptionCollection:
    """Decode all SEQDESC custom waveforms from an MRD stream.

    Non-SEQDESC waveforms (for example ECG or respiratory traces) are ignored.
    The input order is irrelevant.
    """

    parameters: SequenceParameters | None = None
    events_by_subsequence: dict[int, tuple[float, tuple[SequenceEvent, ...]]] = {}
    rf_by_subsequence: dict[int, dict[int, RfDefinition]] = {}
    shims_by_subsequence: dict[int, dict[int, ShimDefinition]] = {}

    for waveform in waveforms:
        waveform_id = int(getattr(waveform, "waveform_id", -1))
        if waveform_id not in _SEQDESC_IDS:
            continue
        words = _payload_words(waveform)
        if waveform_id == WAVEFORM_ID_SEQDESC_HEADER:
            if parameters is not None:
                raise ValueError("duplicate SEQDESC header waveform")
            parameters = _decode_header(words)
        elif waveform_id == WAVEFORM_ID_SEQDESC_EVENTS:
            subseq, tr_us, events = _decode_events(words)
            if subseq in events_by_subsequence:
                raise ValueError(
                    f"duplicate SEQDESC event waveform for subsequence {subseq}"
                )
            events_by_subsequence[subseq] = (tr_us, events)
        elif waveform_id == WAVEFORM_ID_SEQDESC_RF_SHAPES:
            subseq, definitions = _decode_rf_definitions(words)
            if subseq in rf_by_subsequence:
                raise ValueError(
                    f"duplicate SEQDESC RF waveform for subsequence {subseq}"
                )
            rf_by_subsequence[subseq] = definitions
        else:
            subseq, shims = _decode_shims(words)
            if subseq in shims_by_subsequence:
                raise ValueError(
                    f"duplicate SEQDESC shim waveform for subsequence {subseq}"
                )
            shims_by_subsequence[subseq] = shims

    if parameters is None:
        raise ValueError("SEQDESC header waveform 999 is missing")
    missing = set(range(parameters.num_subsequences)) - set(events_by_subsequence)
    if missing:
        raise ValueError(
            f"SEQDESC event waveforms missing for subsequences {sorted(missing)}"
        )

    subsequences = {}
    for subseq, (tr_us, events) in events_by_subsequence.items():
        definitions = rf_by_subsequence.get(subseq, {})
        referenced = {
            event.rf_definition_id for event in events if event.type is EventType.RF
        }
        missing_rf = referenced - set(definitions)
        if missing_rf:
            raise ValueError(
                f"subsequence {subseq} references missing RF definitions {sorted(missing_rf)}"
            )
        subsequences[subseq] = SequenceDescription(
            subsequence_index=subseq,
            tr_duration_us=tr_us,
            events=events,
            rf_definitions=definitions,
            shim_definitions=shims_by_subsequence.get(subseq, {}),
        )
    return SequenceDescriptionCollection(
        parameters=parameters, subsequences=subsequences
    )


def _payload_words(waveform: Any) -> np.ndarray:
    data = np.asarray(getattr(waveform, "data", waveform))
    if data.dtype != np.uint32:
        raise TypeError(f"SEQDESC payload must be uint32, got {data.dtype}")
    channels = int(getattr(waveform, "channels", 1))
    if channels != 1:
        raise ValueError(f"SEQDESC waveform must have one channel, got {channels}")
    return np.ascontiguousarray(data).reshape(-1)


def _float(word: np.uint32) -> float:
    return float(np.asarray([word], dtype="<u4").view("<f4")[0])


def _take_float(words: np.ndarray, cursor: int) -> tuple[float, int]:
    if cursor >= words.size:
        raise ValueError("truncated SEQDESC payload")
    return _float(words[cursor]), cursor + 1


def _take_int(words: np.ndarray, cursor: int) -> tuple[int, int]:
    if cursor >= words.size:
        raise ValueError("truncated SEQDESC payload")
    return int(words[cursor]), cursor + 1


def _decode_header(words: np.ndarray) -> SequenceParameters:
    if words.size != 6:
        raise ValueError(f"SEQDESC header has {words.size} words, expected 6")
    return SequenceParameters(
        num_subsequences=int(words[0]),
        min_te_us=_float(words[1]),
        min_tr_us=_float(words[2]),
        max_tr_us=_float(words[3]),
        max_flip_angle_deg=_float(words[4]),
        total_scan_time_us=_float(words[5]),
    )


def _decode_events(
    words: np.ndarray,
) -> tuple[int, float, tuple[SequenceEvent, ...]]:
    if words.size < 3:
        raise ValueError("truncated SEQDESC event header")
    subseq = int(words[0])
    tr_us = _float(words[1])
    count = int(words[2])
    expected = 3 + 9 * count
    if words.size != expected:
        raise ValueError(f"SEQDESC events have {words.size} words, expected {expected}")
    events = []
    cursor = 3
    last_time = -np.inf
    for _ in range(count):
        event_type = EventType(int(words[cursor]))
        timestamp_us = _float(words[cursor + 1])
        params = tuple(_float(word) for word in words[cursor + 2 : cursor + 9])
        if timestamp_us < last_time:
            raise ValueError("SEQDESC events are not timestamp ordered")
        events.append(SequenceEvent(event_type, timestamp_us, params))
        last_time = timestamp_us
        cursor += 9
    return subseq, tr_us, tuple(events)


def _decode_rf_definitions(
    words: np.ndarray,
) -> tuple[int, dict[int, RfDefinition]]:
    cursor = 0
    subseq, cursor = _take_int(words, cursor)
    count, cursor = _take_int(words, cursor)
    definitions = {}
    for _ in range(count):
        rf_id, cursor = _take_int(words, cursor)
        bandwidth, cursor = _take_float(words, cursor)
        num_bands, cursor = _take_int(words, cursor)
        offsets = []
        for _band in range(8):
            value, cursor = _take_float(words, cursor)
            offsets.append(value)
        band_bandwidth, cursor = _take_float(words, cursor)
        b1sq, cursor = _take_float(words, cursor)
        mag_nunc, cursor = _take_int(words, cursor)
        mag_n, cursor = _take_int(words, cursor)
        has_phase, cursor = _take_int(words, cursor)
        phase_header = None
        if has_phase:
            phase_nunc, cursor = _take_int(words, cursor)
            phase_n, cursor = _take_int(words, cursor)
            phase_header = (phase_nunc, phase_n)
        has_time, cursor = _take_int(words, cursor)
        time_header = None
        if has_time:
            time_nunc, cursor = _take_int(words, cursor)
            time_n, cursor = _take_int(words, cursor)
            time_header = (time_nunc, time_n)

        def take_shape(nunc: int, n: int) -> RfShape:
            nonlocal cursor
            if n < 0 or cursor + n > words.size:
                raise ValueError("truncated SEQDESC RF shape")
            samples = words[cursor : cursor + n].view("<f4").copy()
            cursor += n
            return RfShape(nunc, samples)

        magnitude = take_shape(mag_nunc, mag_n)
        phase = take_shape(*phase_header) if phase_header else None
        time = take_shape(*time_header) if time_header else None
        if rf_id in definitions:
            raise ValueError(f"duplicate RF definition {rf_id} in subsequence {subseq}")
        definitions[rf_id] = RfDefinition(
            id=rf_id,
            bandwidth_hz=bandwidth,
            num_bands=num_bands,
            band_frequency_offsets_hz=tuple(offsets),
            band_bandwidth_hz=band_bandwidth,
            total_b1sq_power=b1sq,
            magnitude=magnitude,
            phase=phase,
            time=time,
        )
    if cursor != words.size:
        raise ValueError(f"SEQDESC RF payload has {words.size - cursor} trailing words")
    return subseq, definitions


def _decode_shims(
    words: np.ndarray,
) -> tuple[int, dict[int, ShimDefinition]]:
    cursor = 0
    subseq, cursor = _take_int(words, cursor)
    count, cursor = _take_int(words, cursor)
    shims = {}
    for _ in range(count):
        shim_id, cursor = _take_int(words, cursor)
        channels, cursor = _take_int(words, cursor)
        magnitudes = []
        phases = []
        for _channel in range(channels):
            value, cursor = _take_float(words, cursor)
            magnitudes.append(value)
        for _channel in range(channels):
            value, cursor = _take_float(words, cursor)
            phases.append(value)
        shims[shim_id] = ShimDefinition(shim_id, tuple(magnitudes), tuple(phases))
    if cursor != words.size:
        raise ValueError(
            f"SEQDESC shim payload has {words.size - cursor} trailing words"
        )
    return subseq, shims
