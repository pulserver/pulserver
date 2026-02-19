"""Test intermediate Sequence representation."""

from copy import deepcopy

import pytest

import numpy as np
import pypulseq as pp

from pulserver import Sequence
from pulserver._core._ceq import PulseqRF, PulseqGrad
from pulserver._core._sequence import _pp_rotate

from helpers import are_equal

sample_rf = pp.make_arbitrary_rf(
    signal=np.asarray([1.0, 2.0, 3.0]), flip_angle=1.0, no_signal_scaling=True
)
sample_grad_trap = pp.make_trapezoid("z", area=1e3)


@pytest.fixture
def system():
    return pp.Opts.default


@pytest.fixture
def pulseq_rf():
    return PulseqRF.from_struct(sample_rf)


@pytest.fixture
def pulseq_grad_trap():
    return PulseqGrad.from_struct(sample_grad_trap)


# Test for Sequence Initialization
def test_sequence_initialization(system):
    seq = Sequence(system, platform="pulseq")
    assert seq._system == system
    assert seq._format == "siemens"
    assert seq._block_library == {"delay": {}}


# Test event registration
def test_register_block(system):
    seq = Sequence(system, platform="pulseq")
    seq.register_block(name="test_block", rf=sample_rf)

    assert "test_block" in seq._block_library
    assert are_equal(seq._block_library["test_block"]["rf"], sample_rf)


# Test section creation
def test_section(system):
    seq = Sequence(system, platform="pulseq")
    seq.section("section_1")
    assert seq._sections_edges == [0]  # No events added yet


# Test block addition for pulseq format
def test_add_block_pulseq(system):
    seq = Sequence(system, platform="pulseq")
    seq.register_block(name="test_block", rf=sample_rf)
    seq.add_block("test_block", rf_amp=1.5, rf_phase=0.1, rf_freq=50.0)

    assert len(seq._sequence.block_events) == 1
    assert np.isclose(
        seq._sequence.get_block(1).rf.signal, sample_rf.signal * 1.5
    ).all()
    assert seq._sequence.get_block(1).rf.phase_offset == 0.1
    assert seq._sequence.get_block(1).rf.freq_offset == 50.0


# Test block addition for custom format (non-pulseq)
def test_add_block_custom_format(system):
    seq = Sequence(system, platform="toppe")  # Non-pulseq format
    seq.register_block(name="test_block", rf=sample_rf)
    seq.add_block("test_block", rf_amp=1.5, rf_phase=0.1, rf_freq=50.0)

    assert len(seq._loop) == 1
    assert seq._loop[0][2] == 1.5  # rf_amp
    assert seq._loop[0][3] == 0.1  # rf_phase
    assert seq._loop[0][4] == 50.0  # rf_freq


# Test block rotation in add_block
def test_block_rotation(system):
    seq = Sequence(system, platform="toppe")
    seq.register_block(name="test_block", gz=sample_grad_trap),

    rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    seq.add_block("test_block", rotmat=rotation_matrix)

    assert (
        seq._loop[0][11] == rotation_matrix[0, 1]
    )  # Check if rotation matrix is used correctly
    assert seq._loop[0][18] == rotation_matrix[2, 2]  # Last row of rotmat


# Test sequence export in pulseq format
def test_export_pulseq_format(system):
    seq = Sequence(system, platform="pulseq")
    result = seq.build()
    assert isinstance(result, pp.Sequence)  # Export should return a PyPulseq Sequence


# Test sequence export in custom format
def test_export_custom_format(system):
    seq = Sequence(system, platform="toppe")  # Non-pulseq format
    seq.register_block(name="test_block", rf=sample_rf)
    seq.add_block("test_block")
    result = seq.build().export("bytes")
    assert isinstance(
        result, bytes
    )  # Exporting in 'bytes' format should return bytes data


# Test _pp_rotate helper function
def test_pp_rotate():
    grad = {
        "gx": np.array([1, 2, 3]),
        "gy": np.array([4, 5, 6]),
        "gz": np.array([7, 8, 9]),
    }
    rotation_matrix = np.eye(3)  # Identity matrix for no rotation
    rotated_grad = _pp_rotate(deepcopy(grad), rotation_matrix)

    assert np.allclose(rotated_grad["gx"], grad["gx"])
    assert np.allclose(rotated_grad["gy"], grad["gy"])
    assert np.allclose(rotated_grad["gz"], grad["gz"])


# Test _pp_rotate with actual rotation
def test_pp_rotate_with_rotation():
    grad = {"gx": np.array([1, 0]), "gy": np.array([0, 1]), "gz": np.array([0, 0])}
    rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 degree rotation
    rotated_grad = _pp_rotate(deepcopy(grad), rotation_matrix)

    assert np.allclose(rotated_grad["gx"], [0, -1])  # Rotated x-axis
    assert np.allclose(rotated_grad["gy"], [1, 0])  # y-axis unchanged
