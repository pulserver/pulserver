"""The pTx tour of the generated prescan-only subsequence: every channel driven as hard as the scan drives it."""

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import ir
from pulserver.ir import Prescan, WaveBudget

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _ptx(lead, scale):
    """A two-channel pulse led by ``lead``, the other at a third of it, at ``scale``."""
    signal = np.tile(np.hanning(100) * 60.0 * scale, (2, 1)).astype(complex)
    signal[lead] *= 3.0 * np.exp(0.3j)
    return pp.make_ptx_pulse(
        signal, dwell=SYSTEM.rf_raster_time, system=SYSTEM, use="excitation"
    )


def _converted(tmp_path, seq):
    path = tmp_path / "tour.seq"
    seq.write(str(path))
    read = pp.io.read(str(path))
    designed = [(path, read)]
    generated = ir.receive_calibration_sequence([read], SYSTEM)
    if generated is not None:
        designed.append((tmp_path / "prescan_calibration.seq", generated))
    ir.convert(path, SYSTEM, designed=designed)
    return path, generated


def _alike(count=5):
    """Repetitions of a pTx preparation led by channel 0, then a pTx excitation led by channel 1 and its readout."""
    adc = pp.make_adc(64, duration=2e-3, system=SYSTEM)
    seq = pp.Sequence(SYSTEM)
    for _ in range(count):
        seq.add_block(_ptx(0, 0.5))
        seq.add_block(pp.make_delay(2e-3))
        seq.add_block(_ptx(1, 0.8))
        seq.add_block(adc)
        seq.add_block(pp.make_delay(5e-3))
    seq.set_definition("NumGainCalibrationReadouts", 1)
    return seq


def _prescan(path):
    cal = ir.receive_calibration(path)
    return cal, Prescan(subsequence=cal["subsequence"])


def test_a_single_channel_sequence_that_repeats_alike_has_no_generated_subsequence(
    tmp_path,
):
    seq = pp.Sequence(SYSTEM)
    for _ in range(3):
        seq.add_block(pp.make_block_pulse(0.2, duration=1e-3, system=SYSTEM))
        seq.add_block(pp.make_adc(64, duration=2e-3, system=SYSTEM))
        seq.add_block(pp.make_delay(5e-3))

    _, generated = _converted(tmp_path, seq)

    assert generated is None


def test_a_ptx_sequence_that_repeats_alike_prescans_after_a_tour_of_its_channels(
    tmp_path,
):
    path, generated = _converted(tmp_path, _alike())
    cal, prescan = _prescan(path)
    blocks = ir.playout(path, WaveBudget(10**6, 4.0), prescan=prescan)["blocks"]
    lead = blocks["subsequence"] == cal["lead"]

    assert generated is not None
    assert cal["subsequence"] == 0
    assert cal["lead"] == 1
    assert lead[: lead.sum()].all()
    assert blocks["adc"][lead].sum() == 0
    assert blocks["adc"][~lead].sum() == 1
    np.testing.assert_allclose(
        ir.channel_drives(path, prescan), ir.channel_drives(path)
    )


def test_the_tour_drives_each_channel_where_the_scan_drives_it_hardest(tmp_path):
    adc = pp.make_adc(64, duration=2e-3, system=SYSTEM)
    seq = pp.Sequence(SYSTEM)
    for prepare, excite in zip(
        [0.2, 0.5, 0.3, 1.0, 0.4], [0.6, 1.0, 0.8, 0.4, 0.2], strict=True
    ):
        seq.add_block(_ptx(0, prepare))
        seq.add_block(pp.make_delay(2e-3))
        seq.add_block(_ptx(1, excite))
        seq.add_block(adc)
        seq.add_block(pp.make_delay(5e-3))
    path, _ = _converted(tmp_path, seq)
    _, prescan = _prescan(path)

    np.testing.assert_allclose(
        ir.channel_drives(path, prescan), ir.channel_drives(path)
    )


def test_a_single_waveform_in_a_ptx_sequence_is_toured_on_every_channel(tmp_path):
    seq = pp.Sequence(SYSTEM)
    for flip in (0.3, 2.0, 0.6):
        seq.add_block(pp.make_block_pulse(flip, duration=1e-3, system=SYSTEM))
        seq.add_block(_ptx(0, 0.1))
        seq.add_block(pp.make_adc(64, duration=2e-3, system=SYSTEM))
        seq.add_block(pp.make_delay(5e-3))
    path, _ = _converted(tmp_path, seq)
    _, prescan = _prescan(path)
    scan = ir.channel_drives(path)

    assert scan[0] == pytest.approx(scan[1])
    np.testing.assert_allclose(ir.channel_drives(path, prescan), scan)


def test_a_shimmed_pulse_is_toured_where_each_channel_is_driven_hardest(tmp_path):
    seq = pp.Sequence(SYSTEM)
    for weights, flip in [((1.0, 0.2), 1.0), ((0.2, 1.0), 0.8), ((1.0, 0.2), 0.5)]:
        seq.add_block(
            pp.make_block_pulse(flip, duration=1e-3, system=SYSTEM),
            pp.make_rf_shim(np.array(weights, dtype=complex)),
        )
        seq.add_block(pp.make_adc(64, duration=2e-3, system=SYSTEM))
        seq.add_block(pp.make_delay(5e-3))
    path, _ = _converted(tmp_path, seq)
    _, prescan = _prescan(path)

    np.testing.assert_allclose(
        ir.channel_drives(path, prescan), ir.channel_drives(path)
    )


def test_the_scan_plays_no_prescan_only_subsequence(tmp_path):
    path, _ = _converted(tmp_path, _alike())
    cal = ir.receive_calibration(path)
    blocks = ir.playout(path, WaveBudget(10**6, 4.0))["blocks"]

    assert (blocks["subsequence"] != cal["lead"]).all()
