"""The event stream a simulation reads, taken off the sequence the scanner plays."""

import base64
import json
import socket
import threading

import ismrmrd
import numpy as np
import pypulseqpp as pp
import pytest

from pulserver.proxy._seqdesc import (
    MESSAGE_KEY,
    AdcRole,
    EventType,
    RfUse,
    SequenceDescription,
    SequenceEvent,
    as_rows,
    describe,
    is_message,
    message,
)
from pulserver.recon._runtime.connection import Connection

SYSTEM = pp.Opts(B0=3.0)


def flip_of(description, event):
    """The angle an RF event turns through, read back as a simulation reads it."""
    definition = description.rf_definitions[int(event.params[0])]
    envelope = (
        definition.magnitude.samples
        * np.exp(2j * np.pi * definition.phase.samples)
        * float(event.params[2])
    )
    return 360.0 * abs(np.sum(envelope)) * description.rf_raster_time_s


def gre(flip_deg=30.0, samples=64, system=SYSTEM):
    seq = pp.Sequence(system)
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=2e-3,
        slice_thickness=5e-3,
        system=system,
        return_gz=True,
        use="excitation",
    )
    readout = pp.make_trapezoid("x", flat_area=2000, flat_time=2.56e-3, system=SYSTEM)
    adc = pp.make_adc(
        num_samples=samples,
        duration=2.56e-3,
        delay=readout.rise_time,
        system=SYSTEM,
    )
    seq.add_block(rf, gz)
    seq.add_block(gzr)
    seq.add_block(readout, adc)
    return seq


def test_one_row_per_block_in_play_order():
    description = describe(gre())
    assert [EventType(event.type) for event in description.events] == [
        EventType.RF,
        EventType.WAIT,
        EventType.ADC,
    ]
    assert len(description) == 3


@pytest.mark.parametrize("flip_deg", [5.0, 30.0, 90.0])
def test_a_pulse_turns_through_the_angle_it_was_designed_for(flip_deg):
    """The whole point: a simulation reading this must see the sequence's own angle."""
    description = describe(gre(flip_deg))
    assert flip_of(description, description.events[0]) == pytest.approx(
        flip_deg, abs=1e-3
    )


def test_a_pulse_keeps_the_sign_of_its_lobes():
    """A sinc's negative lobes cancel; a magnitude alone would add them.

    Dropping the phase turns a 30 degree pulse into a 48 degree one, which is
    why the definition carries both shapes.
    """
    description = describe(gre(30.0))
    definition = description.rf_definitions[int(description.events[0].params[0])]
    amplitude = float(description.events[0].params[2])
    without_phase = (
        360.0
        * abs(np.sum(definition.magnitude.samples * amplitude))
        * description.rf_raster_time_s
    )
    assert without_phase > 45.0
    assert flip_of(description, description.events[0]) == pytest.approx(30.0, abs=1e-3)


def test_an_excitation_says_it_is_one():
    description = describe(gre())
    assert RfUse(int(description.events[0].params[1])) is RfUse.EXCITATION


def test_a_pulse_is_timed_at_its_centre():
    seq = gre()
    description = describe(seq)
    assert description.events[0].timestamp_us == pytest.approx(
        float(seq.rf_times(compat=False).t[0]) * 1e6
    )


def test_a_readout_is_timed_where_it_passes_the_centre_of_k_space():
    """Not the middle of the window: an asymmetric readout reaches k=0 early."""
    seq = gre()
    description = describe(seq)
    adc = next(e for e in description.events if e.type == EventType.ADC)
    samples = np.asarray(seq.adc_times()[0])
    echo = np.asarray(seq.adc_echoes().echo)[0].min()
    first = int(np.asarray(seq.adc_echoes().first_sample)[0])
    assert adc.timestamp_us == pytest.approx(float(samples[first + echo]) * 1e6)
    assert AdcRole(int(adc.params[0])) is AdcRole.SINGLE


def test_a_pulse_played_under_a_gradient_carries_its_amplitude():
    """A slice-selective pulse turns only the slice, which needs the gradient."""
    description = describe(gre())
    assert float(description.events[0].params[6]) != 0.0


def test_a_pulse_driven_through_a_shim_names_it():
    seq = pp.Sequence(SYSTEM)
    pulse = pp.make_block_pulse(np.pi / 2, duration=1e-3, system=SYSTEM)
    seq.add_block(pulse, pp.make_rf_shim(np.array([1.0, 1j])))
    seq.add_block(pulse)
    description = describe(seq)
    assert [int(event.params[5]) for event in description.events] == [1, 0]


def test_the_block_table_is_read_once_however_many_pulses_play(monkeypatch):
    """Reading it per pulse made a 70,000-readout scan take half an hour."""
    seq = pp.Sequence(SYSTEM)
    pulse = pp.make_block_pulse(np.pi / 18, duration=1e-4, system=SYSTEM)
    for _ in range(50):
        seq.add_block(pulse)
        seq.add_block(pp.make_delay(1e-3))
    reads = {"block_events": 0, "block_shims": 0}
    cls = type(seq)
    events, shims = cls.block_events, cls.block_shims

    def counted_events(self):
        reads["block_events"] += 1
        return events.fget(self)

    def counted_shims(self):
        reads["block_shims"] += 1
        return shims(self)

    monkeypatch.setattr(cls, "block_events", property(counted_events))
    monkeypatch.setattr(cls, "block_shims", counted_shims)
    assert len(describe(seq)) == 100
    assert reads == {"block_events": 1, "block_shims": 1}


def test_the_stream_flattens_to_one_row_per_block():
    description = describe(gre())
    rows = as_rows(description)
    assert rows["type"].shape == (3,)
    assert rows["params"].shape == (3, 7)
    assert rows["type"].tolist() == [EventType.RF, EventType.WAIT, EventType.ADC]
    # A wait carries nothing, and the row is zeros rather than absent.
    assert not rows["params"][1].any()


@pytest.mark.parametrize("raster_s", [1e-6, 2e-6])
def test_a_pulse_integrated_over_its_sample_times_turns_through_its_angle(raster_s):
    """Sample times are in raster steps, which a reader scales by the raster it is told."""
    description = describe(gre(30.0, system=pp.Opts(B0=3.0, rf_raster_time=raster_s)))
    event = description.events[0]
    definition = description.rf_definitions[int(event.params[0])]
    envelope = (
        definition.magnitude.samples
        * np.exp(2j * np.pi * definition.phase.samples)
        * float(event.params[2])
    )
    time_s = definition.time.samples * description.rf_raster_time_s
    assert description.rf_raster_time_s == raster_s
    assert 360.0 * abs(np.trapezoid(envelope, time_s)) == pytest.approx(30.0, rel=1e-2)


def _sent_and_read(text):
    """Send ``text`` and one acquisition over a socket; return what the far end reads."""
    near, far = socket.socketpair()
    far.settimeout(60.0)

    def send():
        sending = Connection(near)
        sending.send(text)
        sending.send(ismrmrd.Acquisition.from_array(np.ones((1, 4), np.complex64)))
        sending.send_close()

    # A socket pair buffers a few kilobytes on macOS, less than the message.
    sender = threading.Thread(target=send, daemon=True)
    with near, far:
        sender.start()
        read = list(Connection(far))
        sender.join()
    return read


def _decoded(text):
    """Read the message back as a receiver does: JSON, then base64 little-endian arrays."""

    def array(packed, dtype):
        return np.frombuffer(base64.b64decode(packed), dtype=dtype)

    subsequences = []
    for each in json.loads(text)[MESSAGE_KEY]:
        rows = {
            "type": array(each["type"], "<i4"),
            "timestamp_us": array(each["timestamp_us"], "<f8"),
            "params": array(each["params"], "<f4").reshape(-1, 7),
        }
        shapes = {
            definition["id"]: {
                name: array(definition[name]["samples"], "<f4")
                for name in ("magnitude", "phase", "time")
            }
            for definition in each["rf_definitions"]
        }
        subsequences.append((each, rows, shapes))
    return subsequences


def test_a_written_stream_reads_back_into_the_rows_it_was_built_from():
    description = describe(gre())
    read = _sent_and_read(message([description]))

    assert is_message(read[0])
    assert isinstance(read[1], ismrmrd.Acquisition)
    ((header, rows, shapes),) = _decoded(read[0])
    expected = as_rows(description)
    for name in ("type", "timestamp_us", "params"):
        np.testing.assert_array_equal(rows[name], expected[name])
    assert header["tr_duration_us"] == description.tr_duration_us
    assert header["rf_raster_time_s"] == description.rf_raster_time_s
    for identifier, definition in description.rf_definitions.items():
        for name in ("magnitude", "phase", "time"):
            np.testing.assert_array_equal(
                shapes[identifier][name],
                getattr(definition, name).samples.astype(np.float32),
            )


def test_an_event_an_hour_into_the_scan_keeps_its_microsecond():
    late_us = 3600e6 + 1.0
    description = SequenceDescription(
        subsequence_index=0,
        tr_duration_us=late_us + 1.0,
        events=(SequenceEvent(EventType.WAIT, late_us),),
        rf_definitions={},
    )
    ((_, rows, _),) = _decoded(message([description]))
    assert rows["timestamp_us"][0] == late_us


def lines(steps=(-2, -1, 0, 1, 2), navigator=False):
    """A spoiled line per repetition, phase-encoded through ``steps``; a navigator before each."""
    seq = pp.Sequence(SYSTEM)
    rf = pp.make_block_pulse(np.deg2rad(10.0), duration=0.2e-3, system=SYSTEM, use="excitation")
    encode = pp.make_trapezoid("y", area=100.0, duration=1e-3, system=SYSTEM)
    readout = pp.make_trapezoid("x", flat_area=2000, flat_time=2.56e-3, system=SYSTEM)
    adc = pp.make_adc(64, duration=2.56e-3, delay=readout.rise_time, system=SYSTEM)
    probe = pp.make_adc(16, duration=0.16e-3, system=SYSTEM)
    nav = pp.make_label(type="SET", label="NAV", value=True)
    image = pp.make_label(type="SET", label="NAV", value=False)
    for step in steps:
        seq.add_block(rf)
        if navigator:
            seq.add_block(probe, nav)
        seq.add_block(pp.scale_grad(encode, step / 2.0))
        seq.add_block(readout, adc, image)
    return seq


def test_a_scan_is_described_by_the_repetition_sampling_the_centre_of_k_space():
    seq = lines()
    description = describe(seq)
    assert len(description) == 3
    start = float(seq.block_durations[1]) + float(seq.block_durations[2]) + float(
        seq.block_durations[3]
    )
    rf = description.events[0]
    assert rf.timestamp_us == pytest.approx(
        (float(seq.rf_times(compat=False).t[2]) - 2 * start) * 1e6
    )
    assert description.tr_duration_us == pytest.approx(start * 1e6)
    # Only the centre line passes k = 0: its readout is the one described.
    adc = description.events[2]
    assert adc.type == EventType.ADC and 0.0 < adc.timestamp_us < description.tr_duration_us


def test_a_navigator_at_the_centre_does_not_choose_the_repetition():
    """A navigator passes k = 0 in every repetition; the first is not the centre's."""
    seq = lines(navigator=True)
    description = describe(seq)
    assert len(description) == 4
    rf_centre = float(seq.rf_times(compat=False).t[2]) * 1e6
    assert description.events[0].timestamp_us == pytest.approx(
        rf_centre - 2 * description.tr_duration_us
    )


def test_a_readout_without_an_echo_is_timed_at_its_first_sample():
    seq = pp.Sequence(SYSTEM)
    seq.add_block(pp.make_block_pulse(np.pi / 18, duration=1e-4, system=SYSTEM))
    seq.add_block(pp.make_adc(32, dwell=10e-6, system=SYSTEM))
    description = describe(seq)
    samples = np.asarray(seq.adc_times()[0])
    assert description.events[1].timestamp_us == pytest.approx(float(samples[0]) * 1e6)


def test_a_kept_series_holds_the_description_and_replays_it_first(tmp_path):
    from pulserver.proxy._queue import DESCRIPTION, QueueFile

    text = message([describe(gre())])
    kept = QueueFile(tmp_path / "series.h5")
    kept.append(text)
    kept.append(ismrmrd.Acquisition.from_array(np.ones((1, 4), np.complex64)))
    kept.close()

    import h5py

    with h5py.File(tmp_path / "series.h5", "r") as stored:
        assert stored["dataset"][DESCRIPTION][()].decode() == text
    replayed = list(kept)
    assert replayed[0] == text
    assert isinstance(replayed[1], ismrmrd.Acquisition)
