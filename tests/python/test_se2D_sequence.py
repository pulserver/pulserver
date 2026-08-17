"""The 2D Cartesian spin echo of the sequence zoo.

Checked through what a reconstruction actually receives, plus the one thing
a spin echo owes beyond a gradient echo: a refocusing pulse at TE/2 whose
k-flip lands the echo on the centre of the readout.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver import UIParam, params
from pulserver.app import se2D_sequence

N_X = 32
N_Y = 16


def design(**kwargs):
    kwargs.setdefault("te", 20e-3)
    kwargs.setdefault("tr", 100e-3)
    kwargs.setdefault("n_acs", 6)
    return se2D_sequence.main(n_x=N_X, n_y=N_Y, **kwargs)


def acquired_lines(seq):
    k_traj_adc, *_ = seq.calculate_kspace()
    labels = seq.evaluate_labels(evolution="adc")
    n_samples = k_traj_adc.shape[1] // len(labels["LIN"])
    ky = k_traj_adc[1].reshape(-1, n_samples)[:, 0]
    return np.rint(ky * seq.get_definition("FOV")[1]).astype(int) + N_Y // 2


def test_the_sequence_is_valid_pulseq():
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


def test_the_refocusing_pulse_sits_at_half_the_echo_time():
    seq = design(te=24e-3)
    _, _, t_excitation, t_refocusing, _ = seq.calculate_kspace()
    assert t_refocusing[0] - t_excitation[0] == pytest.approx(12e-3, abs=1e-5)


def test_the_echo_lands_on_the_centre_of_the_readout():
    """The k-flip in action: kx crosses zero at the centre sample."""
    seq = design()
    k_traj_adc, *_ = seq.calculate_kspace()
    labels = seq.evaluate_labels(evolution="adc")
    n_samples = k_traj_adc.shape[1] // len(labels["LIN"])
    kx = k_traj_adc[0][:n_samples]
    assert int(np.argmin(np.abs(kx))) == n_samples // 2 - 1


def test_the_slice_axis_is_rephased_through_the_refocusing_pulse():
    """The rephaser plays before the 180 and the crushers cancel around it,
    so the acquisition sees kz = 0 without a rephaser of its own after."""
    seq = design()
    k_traj_adc, *_ = seq.calculate_kspace()
    assert float(np.abs(k_traj_adc[2]).max()) == pytest.approx(0.0, abs=1e-6)


def test_the_acquired_lines_are_the_ones_the_sampling_plan_asked_for():
    seq = design(acceleration=2)
    expected = np.asarray(
        pp.calc_sampled_lines(N_Y, 2, 6, order="calibration_first")
    )
    assert np.array_equal(acquired_lines(seq), expected)
    assert np.array_equal(
        seq.evaluate_labels(evolution="adc")["LIN"], acquired_lines(seq)
    )


def test_both_pulses_are_offset_to_the_slice():
    seq = design(n_slices=2, tr=200e-3, slice_thickness=5e-3, slice_gap=1e-3)
    pulses = [
        block.rf
        for index in range(1, seq.num_blocks + 1)
        if (block := seq.get_block(index)).rf is not None
    ]
    excitations = [rf for rf in pulses if rf.use == "e"]
    refocusings = [rf for rf in pulses if rf.use == "r"]
    assert {round(float(rf.freq_offset)) for rf in excitations} == {
        round(266666.6666666666 * 3e-3),
        round(266666.6666666666 * -3e-3),
    }
    # Same SLR design numbers for both pulses, so the same offsets select the
    # same slice through the 180.
    assert {round(float(rf.freq_offset)) for rf in refocusings} == {
        round(float(rf.freq_offset)) for rf in excitations
    }


def test_the_refocusing_pulse_keeps_its_cpmg_quarter_turn():
    seq = design()
    refocusing = next(
        block.rf
        for index in range(1, seq.num_blocks + 1)
        if (block := seq.get_block(index)).rf is not None
        and seq.get_block(index).rf.use == "r"
    )
    assert float(refocusing.phase_offset) == pytest.approx(np.pi / 2)


def test_a_te_shorter_than_either_half_is_refused_by_name():
    with pytest.raises(ValueError, match="TE"):
        design(te=2e-3)


def test_a_feasible_protocol_reports_its_scan_time():
    system = pp.Opts()
    report = se2D_sequence.PLUGIN.validate_protocol(
        system, se2D_sequence.PLUGIN.get_default_protocol(system)
    )
    assert report["valid"] is True
    assert report["duration"] > 0


def test_an_impossible_echo_time_is_refused_through_the_protocol():
    system = pp.Opts()
    protocol = se2D_sequence.PLUGIN.get_default_protocol(system)
    params.set_protocol_value(protocol, UIParam.TE, 1.0)
    report = se2D_sequence.PLUGIN.validate_protocol(system, protocol)
    assert report["valid"] is False
    assert "TE" in report["info"]


def test_a_slice_offset_is_a_frequency_on_both_pulses():
    plain = design()
    shifted = design(fov_offset=(0.0, 0.0, 10e-3))
    plain_pulses = [
        block.rf
        for index in range(1, plain.num_blocks + 1)
        if (block := plain.get_block(index)).rf is not None
    ]
    shifted_pulses = [
        block.rf
        for index in range(1, shifted.num_blocks + 1)
        if (block := shifted.get_block(index)).rf is not None
    ]
    for before, after in zip(plain_pulses, shifted_pulses):
        assert float(after.freq_offset) != pytest.approx(float(before.freq_offset))
    assert shifted._native.num_shapes() == plain._native.num_shapes()
