from __future__ import annotations

import struct

import ismrmrd
import numpy as np
import pytest

from pulserver.mrd import (
    AdcRole,
    EventType,
    RfUse,
    decode_sequence_description,
    decompress_shape,
)


def _u(value: int) -> np.uint32:
    return np.uint32(value)


def _f(value: float) -> np.uint32:
    return np.frombuffer(struct.pack("<f", value), dtype="<u4")[0]


def _waveform(waveform_id: int, words: list[np.uint32]) -> ismrmrd.Waveform:
    return ismrmrd.Waveform.from_array(
        np.asarray([words], dtype=np.uint32),
        waveform_id=waveform_id,
    )


def _resources(*, second_adc_echo: bool = False) -> list[ismrmrd.Waveform]:
    header = [_u(1), _f(5000), _f(10000), _f(10000), _f(90), _f(10000)]
    events = [_u(0), _f(10000), _u(3)]
    events += [_u(EventType.RF), _f(0)]
    events += [
        _f(0),
        _f(RfUse.EXCITATION),
        _f(125000),
        _f(0),
        _f(0),
        _f(-1),
        _f(0),
    ]
    events += [_u(EventType.ADC), _f(5000)]
    events += [
        _f(AdcRole.ECHO_CENTER),
        _f(0),
        _f(1),
        _f(0),
        _f(0),
        _f(0),
        _f(0),
    ]
    events += [_u(EventType.ADC), _f(7000)]
    events += [
        _f(AdcRole.NON_CENTER),
        _f(0),
        _f(int(second_adc_echo)),
        _f(0),
        _f(0),
        _f(0),
        _f(0),
    ]

    # One uncompressed RF definition with magnitude [0, 1, 1, 0].
    rf = [_u(0), _u(1)]
    rf += [_u(0), _f(1000), _u(1)]
    rf += [_f(0)] * 8
    rf += [_f(1000), _f(2e-6)]
    rf += [_u(4), _u(4), _u(0), _u(0)]
    rf += [_f(0), _f(1), _f(1), _f(0)]
    shims = [_u(0), _u(0)]
    return [
        _waveform(1002, rf),
        _waveform(999, header),
        _waveform(1005, shims),
        _waveform(1000, events),
    ]


def test_decode_sequence_description_and_echo_flag():
    decoded = decode_sequence_description(_resources())
    description = decoded.subsequence()

    assert decoded.parameters.num_subsequences == 1
    assert description.tr_duration_us == pytest.approx(10000)
    assert description.events[0].rf_use is RfUse.EXCITATION
    assert description.events[1].is_echo is True
    assert description.events[2].is_echo is False
    assert description.events[2].adc_role is AdcRole.NON_CENTER
    flip, phase = description.rf_definitions[0].flip_angle(125000)
    assert flip == pytest.approx(np.pi / 2)
    assert phase == pytest.approx(0)


def test_decoder_ignores_unrelated_waveforms_and_rejects_missing_header():
    unrelated = _waveform(7, [_u(123)])
    decoded = decode_sequence_description([unrelated, *_resources()])
    assert len(decoded.subsequences) == 1
    with pytest.raises(ValueError, match="header"):
        decode_sequence_description(
            [waveform for waveform in _resources() if waveform.waveform_id != 999]
        )


def test_decompress_shape_delta_rle():
    # Deltas [1, 1, 1, 1, -1] are packed as [1, 1, 2, -1].
    result = decompress_shape(np.array([1, 1, 2, -1], np.float32), 5)
    np.testing.assert_allclose(result, [1, 2, 3, 4, 3])
