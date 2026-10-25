"""The receive-gain calibration: the scan's own start where it repeats alike, a generated prescan-only subsequence otherwise."""

import math

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import ir
from pulserver.ir import Prescan, WaveBudget

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _repetition(seq, flip_deg, thickness, delay=5e-3, phase_area=0.0):
    rf, gz, gzr = pp.make_sinc_pulse(
        math.radians(flip_deg),
        duration=2e-3,
        slice_thickness=thickness,
        time_bw_product=4,
        system=SYSTEM,
        return_gz=True,
        use="excitation",
    )
    seq.add_block(rf, gz)
    if phase_area:
        seq.add_block(
            gzr, pp.make_trapezoid("y", area=phase_area, duration=1e-3, system=SYSTEM)
        )
    else:
        seq.add_block(gzr)
    seq.add_block(pp.make_adc(64, duration=2e-3, system=SYSTEM))
    seq.add_block(pp.make_delay(delay))


def _periodic(flip_deg, thickness, count=4, phase=False):
    seq = pp.Sequence(SYSTEM)
    for k in range(count):
        area = (k - count // 2 + 0.5) * 20.0 if phase else 0.0
        _repetition(seq, flip_deg, thickness, phase_area=area)
    seq.set_definition("NumGainCalibrationReadouts", 1)
    return seq


def _train(flips):
    seq = pp.Sequence(SYSTEM)
    for flip in flips:
        _repetition(seq, flip, 5e-3)
    return seq


def _flips(seq):
    flips = seq.rf_flip_angles()
    rf = np.asarray(seq.libraries().blocks)[:, 0]
    return [abs(float(flips[i - 1])) for i in rf if i > 0]


def _converted(tmp_path, sequences):
    paths = [tmp_path / f"part{i}.seq" for i in range(len(sequences))]
    for path, seq, following in zip(paths, sequences, [*paths[1:], None], strict=True):
        if following is not None:
            seq.set_definition("NextSequence", following.name)
        seq.write(str(path))
    designed = [(path, pp.io.read(str(path))) for path in paths]
    calibration = ir.receive_calibration_sequence([s for _, s in designed], SYSTEM)
    if calibration is not None:
        designed.append((tmp_path / "prescan_calibration.seq", calibration))
    ir.convert(designed[0][0], SYSTEM, designed=designed)
    return designed[0][0], calibration


def test_a_single_sequence_that_repeats_alike_generates_nothing(tmp_path):
    path, calibration = _converted(tmp_path, [_periodic(30, 5e-3)])
    cal = ir.receive_calibration(path)

    assert calibration is None
    assert not cal["generated"]
    assert cal["subsequence"] == 0


def test_a_chain_calibrates_on_the_thick_slab_over_the_strong_thin_slice(tmp_path):
    path, calibration = _converted(tmp_path, [_periodic(90, 3e-3), _periodic(15, 0.1)])
    cal = ir.receive_calibration(path)

    assert calibration is not None
    assert _flips(calibration) == pytest.approx([15, 15], abs=0.1)
    assert calibration.get_definition("NumGainCalibrationReadouts") == [2]
    assert cal["generated"]
    assert cal["subsequence"] == 2
    assert cal["readouts"] == 2


def test_a_flip_angle_train_calibrates_on_its_largest_flip_at_its_own_amplitude(
    tmp_path,
):
    _, calibration = _converted(tmp_path, [_train([10, 35, 80, 20, 55])])

    assert calibration is not None
    assert _flips(calibration) == pytest.approx([80, 80], abs=0.1)


def test_a_sequence_without_readouts_generates_nothing():
    seq = pp.Sequence(SYSTEM)
    seq.add_block(pp.make_block_pulse(0.3, duration=1e-3, system=SYSTEM))
    seq.add_block(pp.make_delay(5e-3))

    assert ir.receive_calibration_sequence([seq], SYSTEM) is None


def test_the_calibration_zeroes_the_gradients_that_vary_across_repetitions():
    calibration = ir.receive_calibration_sequence(
        [_periodic(90, 3e-3), _periodic(15, 0.1, phase=True)], SYSTEM
    )
    phase = [
        calibration.get_block(i).gy
        for i in range(1, calibration.num_blocks + 1)
        if calibration.get_block(i).gy is not None
    ]

    assert all(abs(g.amplitude) == 0 for g in phase)


def test_the_scan_never_plays_the_calibration_and_its_prescan_does(tmp_path):
    path, _ = _converted(tmp_path, [_periodic(90, 3e-3), _periodic(15, 0.1)])
    cal = ir.receive_calibration(path)
    budget = WaveBudget(10**6, 4.0)
    scan = ir.playout(path, budget)["blocks"]
    prescan = ir.playout(path, budget, prescan=Prescan(subsequence=cal["subsequence"]))[
        "blocks"
    ]

    assert (scan["subsequence"] != cal["subsequence"]).all()
    assert (prescan["subsequence"] == cal["subsequence"]).all()
    assert prescan["adc"].sum() == cal["readouts"]


def test_the_calibration_drives_no_channel_beyond_the_scan(tmp_path):
    path, _ = _converted(tmp_path, [_periodic(90, 3e-3), _periodic(15, 0.1)])
    cal = ir.receive_calibration(path)
    prescan = ir.channel_drives(path, Prescan(subsequence=cal["subsequence"]))

    assert (prescan <= ir.channel_drives(path) * (1 + 1e-6)).all()
    assert prescan[0] > 0
