"""The 3D Cartesian spin echo of the sequence zoo.

The volume counterpart of :mod:`test_se2D_sequence`: the spin-echo timing and
k-flip, over the ``(ky, kz)`` traversal the 3D gradient echo established.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver import UIParam, params
from pulserver.app import se3D_sequence

N_X = 16
N_Y = 16
N_Z = 8
SLAB = 64e-3


def design(**kwargs):
    kwargs.setdefault("te", 20e-3)
    kwargs.setdefault("tr", 60e-3)
    kwargs.setdefault("n_acs", 6)
    kwargs.setdefault("n_acs_z", 4)
    return se3D_sequence.main(n_x=N_X, n_y=N_Y, n_z=N_Z, slab_thickness=SLAB, **kwargs)


def kernel(**kwargs):
    kwargs.setdefault("te", 20e-3)
    kwargs.setdefault("tr", 60e-3)
    kwargs.setdefault("n_acs", 6)
    kwargs.setdefault("n_acs_z", 4)
    return se3D_sequence.SE3DKernel(
        pp.Opts(), n_x=N_X, n_y=N_Y, n_z=N_Z, slab_thickness=SLAB, **kwargs
    )


def acquired_pairs(seq):
    k_traj_adc, *_ = seq.calculate_kspace()
    labels = seq.evaluate_labels(evolution="adc")
    n_samples = k_traj_adc.shape[1] // len(labels["LIN"])
    ky = k_traj_adc[1].reshape(-1, n_samples)[:, 0]
    kz_end = k_traj_adc[2].reshape(-1, n_samples)[:, -1]
    fov = seq.get_definition("FOV")
    lines = np.rint(ky * fov[1]).astype(int) + N_Y // 2
    partitions = np.rint(kz_end * fov[2]).astype(int) + N_Z // 2
    return list(zip(lines.tolist(), partitions.tolist(), strict=False))


def test_the_sequence_is_valid_pulseq():
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


def test_the_refocusing_pulse_sits_at_half_the_echo_time():
    seq = design(te=24e-3)
    _, _, t_excitation, t_refocusing, _ = seq.calculate_kspace()
    assert t_refocusing[0] - t_excitation[0] == pytest.approx(12e-3, abs=1e-5)


def test_the_echo_lands_on_the_centre_of_the_readout():
    """Between the two central samples, not at one of them: an even-sampled
    readout straddling the origin has none at k = 0."""
    seq = design()
    k_traj_adc, *_ = seq.calculate_kspace()
    labels = seq.evaluate_labels(evolution="adc")
    n_samples = k_traj_adc.shape[1] // len(labels["LIN"])
    kx = k_traj_adc[0][:n_samples]
    left, right = kx[n_samples // 2 - 1], kx[n_samples // 2]
    step = abs(right - left)
    assert left * right < 0.0
    # The midpoint on the origin to a thousandth of a sample: what survives
    # holding the gradients at the precision the file writes.
    assert abs(left + right) / 2.0 < 1e-3 * step


def test_the_acquired_pairs_are_the_ones_the_plan_asked_for():
    seq = design(acceleration=2)
    assert acquired_pairs(seq) == kernel(acceleration=2).pairs
    labels = seq.evaluate_labels(evolution="adc")
    assert labels["LIN"].tolist() == [line for line, _ in acquired_pairs(seq)]
    assert labels["PAR"].tolist() == [partition for _, partition in acquired_pairs(seq)]


def test_the_calibration_rectangle_is_acquired_before_anything_else():
    plan = kernel(acceleration=2, acceleration_z=2)
    seq = design(acceleration=2, acceleration_z=2)
    first = acquired_pairs(seq)[: plan.n_calibration]
    acs_y = set(range(N_Y // 2 - 3, N_Y // 2 + 3))
    acs_z = set(range(N_Z // 2 - 2, N_Z // 2 + 2))
    assert all(line in acs_y and partition in acs_z for line, partition in first)


def test_a_te_shorter_than_either_half_is_refused_by_name():
    with pytest.raises(ValueError, match="TE"):
        design(te=2e-3)


def test_an_impossible_echo_time_is_refused_through_the_protocol():
    system = pp.Opts()
    protocol = se3D_sequence.PLUGIN.get_default_protocol(system)
    params.set_protocol_value(protocol, UIParam.TE, 1.0)
    report = se3D_sequence.PLUGIN.validate_protocol(system, protocol)
    assert report["valid"] is False
    assert "TE" in report["info"]


def test_the_prescribed_grid_is_declared():
    seq = design()
    assert seq.get_definition("FOV") == pytest.approx([220e-3, 220e-3, SLAB])
    assert seq.get_definition("Matrix") == pytest.approx([N_X, N_Y, N_Z])
    assert seq.get_definition("Name") == "se_3d"


def test_a_slab_offset_is_a_frequency_on_both_pulses_and_costs_no_shape():
    plain = design()
    shifted = design(fov_offset=(0.0, 0.0, 15e-3))
    for index in range(1, plain.num_blocks + 1):
        before, after = plain.get_block(index).rf, shifted.get_block(index).rf
        if before is None:
            continue
        assert float(after.freq_offset) != pytest.approx(float(before.freq_offset))
    assert shifted._native.num_shapes() == plain._native.num_shapes()
