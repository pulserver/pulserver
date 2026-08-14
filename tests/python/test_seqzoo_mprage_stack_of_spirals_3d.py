"""The explicit-arms stack-of-spirals MPRAGE of the sequence zoo.

What this slot exists to stress: every arm is its own waveform, so the
shape library grows with the arm count -- the exact opposite of the
rotation-extension slots -- while the MPRAGE timing contract (TI at the
centre partition, inversions at the outer TR) still holds.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.seqzoo import mprage_stack_of_spirals_3d as slot

N_Z = 4
TI = 200e-3
TR_OUTER = 500e-3


def design(**kwargs):
    kwargs.setdefault("n_arms", 3)
    kwargs.setdefault("ti", TI)
    kwargs.setdefault("tr_outer", TR_OUTER)
    kwargs.setdefault("readout_bandwidth_hz", 125e3)
    return slot.main(n_x=32, n_z=N_Z, slab_thickness=64e-3, **kwargs)


def test_the_sequence_is_valid_pulseq():
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


def test_explicit_arms_grow_the_shape_library():
    """The untemplatable path: a moving waveform is data, and more arms are
    more shapes -- where a rotation-extension scan stays constant."""
    few = design(n_arms=2)
    many = design(n_arms=5)
    assert many._native.num_shapes() > few._native.num_shapes()


def test_no_rotation_extensions_are_used():
    seq = design()
    assert all(
        seq.get_block(index).rotation is None
        for index in range(1, seq.num_blocks + 1)
    )


def test_the_arms_step_by_the_golden_angle():
    seq = design(n_arms=3)
    labels = seq.evaluate_labels(evolution="adc")
    k_adc, *_ = seq.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // len(labels["PAR"])

    # kz = 0 partition of each arm carries the in-plane path undisturbed.
    rows = [
        index
        for index, partition in enumerate(labels["PAR"].tolist())
        if partition == N_Z // 2
    ]
    def arm(row):
        return k_adc[:2, row * n_samples : (row + 1) * n_samples]

    angle = float(np.pi * (3.0 - np.sqrt(5.0)))
    turn = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    assert np.allclose(arm(rows[1]), turn @ arm(rows[0]), atol=1e-4)


def test_every_partition_of_every_arm_is_acquired():
    labels = design(n_arms=3).evaluate_labels(evolution="adc")
    pairs = set(zip(labels["SEG"].tolist(), labels["PAR"].tolist()))
    assert pairs == {(arm, partition) for arm in range(3) for partition in range(N_Z)}


def test_the_centre_partition_is_excited_at_the_inversion_time():
    seq = design()
    inversions, excitations = [], []
    t = 0.0
    for index in range(1, seq.num_blocks + 1):
        block = seq.get_block(index)
        if block.rf is not None:
            centre = t + block.rf.delay + block.rf.center
            (inversions if block.rf.use == "i" else excitations).append(centre)
        t += block.block_duration
    inversions = np.asarray(inversions)
    excitations = np.asarray(excitations)

    first_segment = excitations[
        (excitations > inversions[0]) & (excitations < inversions[1])
    ]
    assert first_segment[N_Z // 2] - inversions[0] == pytest.approx(TI, abs=2e-5)
    assert np.diff(inversions) == pytest.approx(
        np.full(len(inversions) - 1, TR_OUTER), abs=1e-5
    )


def test_the_default_protocol_is_feasible():
    system = pp.Opts()
    report = slot.PLUGIN.validate_protocol(
        system, slot.PLUGIN.get_default_protocol(system)
    )
    assert report["valid"] is True, report["info"]
    assert "explicit arms" in report["info"]
