"""Which repetitions share a segment: those whose blocks play the same pulses."""

import math

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import ir

SYSTEM = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s", B0=3.0)
REPETITIONS = 6


def _readout():
    gx = pp.make_trapezoid("x", flat_area=64 / 0.22, flat_time=3.2e-3, system=SYSTEM)
    rewinder = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=SYSTEM)
    return rewinder, gx


def _adc(samples, gx):
    return pp.make_adc(samples, duration=3.2e-3, delay=gx.rise_time, system=SYSTEM)


def _excitation():
    return pp.make_block_pulse(
        math.pi / 12, duration=1e-3, use="excitation", system=SYSTEM
    )


def _sequence(repetition):
    """``REPETITIONS`` repetitions, the blocks of each listed by ``repetition(index)``."""
    seq = pp.Sequence(SYSTEM)
    for index in range(REPETITIONS):
        for events in repetition(index):
            seq.add_block(*events)
    return seq


def _played(tmp_path, seq):
    path = tmp_path / "scan.seq"
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return ir.play(path, waveforms=True)


def test_repetitions_that_play_different_pulses_each_play_their_own(tmp_path):
    """A sinc and a Gaussian excitation of one duration, in turn."""
    pulses = (
        pp.make_sinc_pulse(math.pi / 6, duration=2e-3, use="excitation", system=SYSTEM),
        pp.make_gauss_pulse(
            math.pi / 6, duration=2e-3, use="excitation", system=SYSTEM
        ),
    )
    rewinder, gx = _readout()
    seq = _sequence(
        lambda index: ((pulses[index % 2],), (rewinder,), (gx, _adc(64, gx)))
    )

    played = _played(tmp_path, seq)

    for block, (start, stop) in enumerate(played["rf_span"]):
        rf = seq.get_block(block + 1).rf
        if rf is None:
            assert start == stop
            continue
        peak = np.abs(rf.signal).max()
        np.testing.assert_allclose(
            played["rf_waveform_hz"][start:stop], rf.signal, rtol=0, atol=1e-5 * peak
        )
    assert played["segment"][0] != played["segment"][3]


def test_repetitions_whose_gradients_differ_in_timing_each_play_their_own(tmp_path):
    """Two spoilers filling blocks of one duration, the second after a delay."""
    rewinder, gx = _readout()
    spoilers = (
        pp.make_trapezoid(
            "z", amplitude=1e4, rise_time=1e-4, flat_time=1.8e-3, system=SYSTEM
        ),
        pp.make_trapezoid(
            "z",
            amplitude=1e4,
            rise_time=1e-4,
            flat_time=1.3e-3,
            delay=5e-4,
            system=SYSTEM,
        ),
    )
    seq = _sequence(
        lambda index: (
            (_excitation(),),
            (rewinder,),
            (gx, _adc(64, gx)),
            (spoilers[index % 2],),
        )
    )

    played = _played(tmp_path, seq)

    for block in range(3, 4 * REPETITIONS, 4):
        spoiler = spoilers[(block // 4) % 2]
        start, stop = played["gradient_span"][block, 2]
        corners = spoiler.delay + np.cumsum(
            [0.0, spoiler.rise_time, spoiler.flat_time, spoiler.fall_time]
        )
        np.testing.assert_allclose(
            played["gradient_time_us"][start:stop], 1e6 * corners, atol=1e-3
        )


def test_a_readout_digitised_two_ways_after_dummy_shots_plays_as_designed(tmp_path):
    """The dummies acquire nothing, so they stand in for either readout."""
    rewinder, gx = _readout()
    readouts = (_adc(64, gx), _adc(128, gx))

    def repetition(index):
        readout = (gx,) if index < 2 else (gx, readouts[index % 2])
        return (_excitation(),), (rewinder,), readout

    seq = _sequence(repetition)

    played = _played(tmp_path, seq)

    designed = [
        0 if seq.get_block(i).adc is None else seq.get_block(i).adc.num_samples
        for i in range(1, 3 * REPETITIONS + 1)
    ]
    assert played["adc_samples"].tolist() == designed


def _pulse_per_shot(shots):
    """A sinc pulse of its own per shot, in blocks of one duration."""
    read = pp.make_trapezoid("x", area=1000, duration=2e-3, system=SYSTEM)
    seq = pp.Sequence(SYSTEM)
    for shot in range(shots):
        seq.add_block(
            pp.make_sinc_pulse(
                math.pi / 6,
                duration=1e-3,
                time_bw_product=2 + 0.05 * shot,
                system=SYSTEM,
            )
        )
        seq.add_block(read)
    return seq


def test_shots_with_a_pulse_of_their_own_each_play_it(tmp_path):
    """pypulseqpp finds the repetition by structure: no definition recurs."""
    seq = _pulse_per_shot(4)

    played = _played(tmp_path, seq)

    for block in range(0, 8, 2):
        start, stop = played["rf_span"][block]
        signal = seq.get_block(block + 1).rf.signal
        np.testing.assert_allclose(
            played["rf_waveform_hz"][start:stop],
            signal,
            rtol=0,
            atol=1e-5 * np.abs(signal).max(),
        )
    unit = ir.summary(tmp_path / "scan.seq", SYSTEM)["subsequences"][0]
    assert unit["tr_size"] == seq.repetition()[0] == 2


def test_arbitrary_gradients_of_two_lengths_in_turn_each_play_at_their_own(tmp_path):
    """In blocks of one duration, so the two differ only in what the gradient plays."""
    rewinder, gx = _readout()
    raster = SYSTEM.grad_raster_time
    lengths = (100, 60)
    arms = [
        pp.make_arbitrary_grad(
            "y",
            5e3 * np.sin(np.pi * (np.arange(count) + 0.5) / count),
            first=0,
            last=0,
            system=SYSTEM,
        )
        for count in lengths
    ]
    seq = _sequence(
        lambda index: (
            (_excitation(),),
            (arms[index % 2], pp.make_delay(max(lengths) * raster)),
            (rewinder,),
            (gx, _adc(64, gx)),
        )
    )

    played = _played(tmp_path, seq)

    for block in range(1, 4 * REPETITIONS, 4):
        arm = arms[(block // 4) % 2]
        start, stop = played["gradient_span"][block, 1]
        area = np.trapezoid(
            played["gradient_waveform_hz_per_m"][start:stop],
            1e-6 * played["gradient_time_us"][start:stop],
        )
        assert area == pytest.approx(arm.area, rel=1e-5)


def test_more_pulses_at_one_position_than_a_segment_holds_prepared_are_refused(
    tmp_path,
):
    seq = _pulse_per_shot(65)
    path = tmp_path / "scan.seq"
    seq.write(str(path))

    with pytest.raises(ValueError, match="too many distinct pulse patterns"):
        ir.convert(path, SYSTEM)
