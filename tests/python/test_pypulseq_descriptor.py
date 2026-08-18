"""``Sequence.sequence_descriptor()``: the recon's own description, design-side.

The point of these tests is that there is **one** descriptor type and one
derivation of it. ``pulserver.recon.simulation.decode_sequence_description``
builds a :class:`~pulserver.recon.simulation.SequenceDescription` from the
description a reconstruction consumes; ``Sequence.sequence_descriptor()`` builds
one from a sequence that has not been written yet. Both come out of
``pulseg_get_sequence_description`` in the C core. If they could drift, a
simulation driven from a design script and a simulation driven from a scan
would stop being comparable, which is the whole reason to have either.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.recon.simulation import (
    AdcRole,
    EventType,
    RfUse,
    SequenceDescription,
)

FLIP_DEGREES = 30.0
REFOCUS_DEGREES = 180.0


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=30,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


def _gre(system, *, repetitions=8, use="excitation", flip=FLIP_DEGREES):
    """A plain gradient echo: one RF, one readout, repeated."""
    seq = pp.Sequence(system=system)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf, gz, gzr = pp.make_sinc_pulse(
            flip_angle=np.deg2rad(flip),
            duration=2e-3,
            slice_thickness=3e-3,
            apodization=0.5,
            time_bw_product=4,
            system=system,
            return_gz=True,
            use=use,
        )
        gx = pp.make_trapezoid(
            channel="x", flat_area=128 / 0.256, flat_time=2.56e-3, system=system
        )
        adc = pp.make_adc(
            num_samples=128, duration=gx.flat_time, delay=gx.rise_time, system=system
        )
        pre = pp.make_trapezoid(
            channel="x", area=-gx.area / 2, duration=1e-3, system=system
        )
        for _ in range(repetitions):
            seq.add_block(rf, gz)
            seq.add_block(pre, gzr)
            seq.add_block(gx, adc)
    return seq


def test_it_returns_the_recons_own_type(system):
    """Not a parallel dataclass -- the one `recon.simulation` already consumes."""
    description = _gre(system).sequence_descriptor()
    assert isinstance(description, SequenceDescription)
    assert description.__class__.__module__ == "pulserver.recon._seqdesc"


def test_one_event_per_block_of_the_repetition_time(system):
    seq = _gre(system)
    description = seq.sequence_descriptor()

    assert len(description.events) == 3
    assert [event.type for event in description.events] == [
        EventType.RF,
        EventType.WAIT,
        EventType.ADC,
    ]
    # The TR is the three blocks it is made of, not the whole scan.
    assert description.tr_duration_us == pytest.approx(
        seq.duration()[0] / 8 * 1e6, rel=1e-6
    )


def test_the_rf_row_carries_its_use_and_recovers_the_designed_flip_angle(system):
    """The flip is not stored; it is the pulse envelope times the amplitude.
    Recovering the number the sequence was designed with is what says the
    envelope, the amplitude and the time base all crossed intact."""
    description = _gre(system).sequence_descriptor()
    (rf_event,) = [event for event in description.events if event.type is EventType.RF]

    assert rf_event.rf_use is RfUse.EXCITATION
    definition = description.rf_definitions[rf_event.rf_definition_id]
    flip, _phase = definition.flip_angle(
        rf_event.rf_amplitude_hz, rf_raster_time_s=pp.Opts().rf_raster_time
    )
    assert np.rad2deg(flip) == pytest.approx(FLIP_DEGREES, rel=1e-4)


def test_a_wrong_envelope_would_fail_that_check(system):
    """Mutation check on the test above -- perturbing the amplitude by a
    percent has to move the recovered flip angle out of tolerance."""
    description = _gre(system).sequence_descriptor()
    (rf_event,) = [event for event in description.events if event.type is EventType.RF]
    definition = description.rf_definitions[rf_event.rf_definition_id]

    flip, _ = definition.flip_angle(
        rf_event.rf_amplitude_hz * 1.01, rf_raster_time_s=pp.Opts().rf_raster_time
    )
    assert np.rad2deg(flip) != pytest.approx(FLIP_DEGREES, rel=1e-4)


@pytest.mark.parametrize(
    ("use", "expected"),
    [
        ("excitation", RfUse.EXCITATION),
        ("refocusing", RfUse.REFOCUSING),
        ("inversion", RfUse.INVERSION),
        ("saturation", RfUse.SATURATION),
        ("preparation", RfUse.PREPARATION),
        ("other", RfUse.OTHER),
    ],
)
def test_every_pulseq_rf_use_survives_into_the_descriptor(system, use, expected):
    """PREPARATION and OTHER used to raise ValueError at decode: the enum
    stopped at SATURATION while the parser produced all seven."""
    description = _gre(system, use=use).sequence_descriptor()
    (rf_event,) = [event for event in description.events if event.type is EventType.RF]
    assert rf_event.rf_use is expected


def test_the_adc_row_is_the_echo_and_says_where_k_is_zero(system):
    description = _gre(system).sequence_descriptor()
    (adc_event,) = description.adc_events

    assert adc_event.adc_role is AdcRole.SINGLE
    assert adc_event.is_echo is True
    # The k-zero timestamp lands inside the repetition time, after the pulse.
    rf_time = next(e.timestamp_us for e in description.events if e.type is EventType.RF)
    assert rf_time < adc_event.timestamp_us < description.tr_duration_us


def test_the_echo_time_agrees_with_the_trajectorys_own_answer(system):
    """Two independent derivations of the same instant: the descriptor's
    k-zero timestamp, and the ADC sample the k-space walk calls the centre."""
    seq = _gre(system)
    description = seq.sequence_descriptor()
    (adc_event,) = description.adc_events

    centres = seq.adc_times(compat=False).echo_center_time
    # The descriptor is TR-relative, the trajectory absolute; compare the
    # offset of the first readout's centre within its own repetition.
    tr_seconds = description.tr_duration_us * 1e-6
    within_tr = centres[0] % tr_seconds
    assert within_tr == pytest.approx(
        adc_event.timestamp_us * 1e-6, abs=2 * system.grad_raster_time
    )


def test_it_does_not_decompress_gradient_waveforms(system, monkeypatch):
    """A simulation wants flip angles and timings; making it pay for the
    waveform decompression it would discard is the one mistake available."""
    seq = _gre(system)
    seq.sequence_descriptor()  # warm the structure cache

    called = []
    monkeypatch.setattr(
        type(seq), "waveforms", lambda self, *a, **k: called.append(1) or []
    )
    monkeypatch.setattr(
        type(seq), "get_gradients", lambda self, *a, **k: called.append(1) or []
    )
    seq.sequence_descriptor()
    assert called == []


def test_it_is_cached_against_the_revision_and_rebuilt_after_a_change(system):
    """Cached so repeated asking is cheap, invalidated so it cannot go stale."""
    seq = _gre(system, repetitions=4)

    seq.sequence_descriptor()
    cached = seq._structure
    seq.sequence_descriptor()
    assert seq._structure is cached, "a second call rebuilt the structure"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seq.add_block(pp.make_delay(5e-3))

    seq.sequence_descriptor()
    assert seq._structure is not cached, (
        "a mutated sequence answered from the old structure"
    )


def test_the_scan_parameters_come_across(system):
    parameters = _gre(system).sequence_parameters()
    assert parameters["num_subseqs"] == 1
    assert parameters["min_tr_us"] == pytest.approx(parameters["max_tr_us"])
    assert parameters["total_scan_time_us"] > parameters["max_tr_us"]
