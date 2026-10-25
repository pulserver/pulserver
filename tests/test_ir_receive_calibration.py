"""The receive-gain calibration: the scan's own start where it repeats, an excitation's span otherwise."""

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import ir
from pulserver.ir import Prescan, WaveBudget

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _excitation(flip_deg, thickness):
    rf, gz, _ = pp.make_sinc_pulse(
        np.deg2rad(flip_deg),
        duration=2e-3,
        slice_thickness=thickness,
        time_bw_product=4,
        system=SYSTEM,
        return_gz=True,
        use="excitation",
    )
    return rf, gz


def _repetition(seq, flip_deg, thickness, delay=5e-3):
    rf, gz = _excitation(flip_deg, thickness)
    seq.add_block(rf, gz)
    seq.add_block(pp.make_trapezoid("z", area=-gz.area / 2, system=SYSTEM))
    seq.add_block(pp.make_adc(64, duration=2e-3, system=SYSTEM))
    seq.add_block(pp.make_delay(delay))


def _periodic(flip_deg, thickness, count=4):
    seq = pp.Sequence(SYSTEM)
    for _ in range(count):
        _repetition(seq, flip_deg, thickness)
    seq.set_definition("NumGainCalibrationReadouts", 1)
    return seq


def _write(tmp_path, sequences):
    names = [f"part{i}.seq" for i in range(len(sequences))]
    for name, following, seq in zip(names, [*names[1:], ""], sequences, strict=True):
        if following:
            seq.set_definition("NextSequence", following)
        seq.write(str(tmp_path / name))
    first = tmp_path / names[0]
    ir.convert(first, SYSTEM)
    return first


def _span(cal):
    return np.arange(cal["first_position"], cal["first_position"] + cal["num_blocks"])


def test_a_single_periodic_sequence_keeps_the_standard_calibration(tmp_path):
    cal = ir.receive_calibration(_write(tmp_path, [_periodic(30, 5e-3)]))

    assert not cal["generated"]
    assert cal["subsequence"] == 0


def test_a_chain_calibrates_on_the_thick_slab_over_the_strong_thin_slice(tmp_path):
    path = _write(tmp_path, [_periodic(90, 3e-3), _periodic(15, 0.1)])
    cal = ir.receive_calibration(path)

    assert cal["generated"]
    assert cal["subsequence"] == 1
    assert cal["readouts"] == 1


def test_a_flip_angle_train_calibrates_on_its_largest_flip(tmp_path):
    seq = pp.Sequence(SYSTEM)
    flips = [10, 35, 80, 20, 55]
    for flip, delay in zip(flips, [4e-3, 6e-3, 5e-3, 7e-3, 3e-3], strict=True):
        _repetition(seq, flip, 5e-3, delay)
    cal = ir.receive_calibration(_write(tmp_path, [seq]))

    assert cal["generated"]
    assert 4 * flips.index(80) in _span(cal)
    assert cal["readouts"] == 1
    assert not cal["balanced"]


def test_a_sequence_without_readouts_has_nothing_to_calibrate(tmp_path):
    seq = pp.Sequence(SYSTEM)
    for flip in (20, 40):
        rf, gz = _excitation(flip, 5e-3)
        seq.add_block(rf, gz)
        seq.add_block(pp.make_delay(5e-3))
    cal = ir.receive_calibration(_write(tmp_path, [seq]))

    assert not cal["generated"]
    assert cal["subsequence"] == -1


@pytest.mark.parametrize("tour", ["none", "played"])
def test_the_generated_calibration_plays_its_span_once_and_acquires_it(tmp_path, tour):
    path = _write(tmp_path, [_periodic(90, 3e-3), _periodic(15, 0.1)])
    cal = ir.receive_calibration(path)
    blocks = ir.playout(
        path,
        WaveBudget(10**6, 4.0),
        prescan=Prescan(tour=tour, calibration="generated"),
    )["blocks"]

    assert (blocks["subsequence"] == cal["subsequence"]).all()
    assert blocks["subsequence"].size == cal["num_blocks"]
    assert blocks["adc"].sum() == cal["readouts"]


def test_the_generated_calibration_drives_no_channel_beyond_the_scan(tmp_path):
    path = _write(tmp_path, [_periodic(90, 3e-3), _periodic(15, 0.1)])
    prescan = ir.channel_drives(path, Prescan(calibration="generated"))

    assert (prescan <= ir.channel_drives(path) * (1 + 1e-6)).all()
    assert prescan[0] > 0
