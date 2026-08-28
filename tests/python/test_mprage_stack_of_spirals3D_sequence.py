"""The stack-of-spirals MPRAGE of the sequence zoo.

Both ways of holding the arms are checked here, because the slot has to
answer for both: turned by a rotation extension, where the shape library is
flat in the arm count, and written out per shot, where it grows with it --
the untemplatable path, a moving waveform being data. Either way the MPRAGE
timing contract (TI at the middle of the train, inversions at the outer TR)
holds.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.app import mprage_stack_of_spirals3D_sequence as slot

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
    more shapes -- where turning one arm stays flat."""
    few = design(n_arms=2, use_rotation_ext=False)
    many = design(n_arms=5, use_rotation_ext=False)
    assert many._native.num_shapes() > few._native.num_shapes()

    turned_few = design(n_arms=2)
    turned_many = design(n_arms=5)
    assert turned_many._native.num_shapes() == turned_few._native.num_shapes()


def test_the_arms_are_turned_by_extension_unless_written_out():
    turned = design()
    assert any(
        turned.get_block(index).rotation is not None
        for index in range(1, turned.num_blocks + 1)
    )

    written = design(use_rotation_ext=False)
    assert all(
        written.get_block(index).rotation is None
        for index in range(1, written.num_blocks + 1)
    )


def test_both_ways_of_holding_the_arms_sample_the_same_k_space():
    """The arms land on the same points however the angle is carried.

    Written out, an arm passes through the shape codec once per angle, and
    what that quantises integrates along the readout. So the two routes are
    compared against the k-space step a sample would have to move by to be a
    different sample -- which is what "the same k-space" means -- rather than
    against the largest k the scan reaches, which is no scale at all.
    """
    fov = 220e-3
    turned = design(fov=fov, partition_angle_offset_deg=6.0)
    written = design(fov=fov, partition_angle_offset_deg=6.0, use_rotation_ext=False)
    k_turned, *_ = turned.calculate_kspace(dense=False)
    k_written, *_ = written.calculate_kspace(dense=False)
    assert np.abs(k_turned - k_written).max() < 1e-3 / fov


def test_a_partition_offset_turns_the_arm_along_kz():
    """Partitions are the outer loop, so with no offset a given arm is played
    at the same angle in every one of them; an offset turns it a little
    further with each, which is the CAIPIRINHA stagger."""
    n_arms = 3

    def angle_of(seq, row):
        k_adc, *_ = seq.calculate_kspace(dense=False)
        n = k_adc.shape[1] // (n_arms * N_Z)
        arm = k_adc[:2, row * n : (row + 1) * n]
        return float(np.degrees(np.arctan2(arm[1, -1], arm[0, -1])))

    # Row `partition * n_arms` is arm 0 of each partition.
    flat = design(n_arms=n_arms)
    held = [angle_of(flat, partition * n_arms) for partition in range(N_Z)]
    assert held == pytest.approx([held[0]] * N_Z, abs=1e-4)

    shifted = design(n_arms=n_arms, partition_angle_offset_deg=6.0)
    staggered = [angle_of(shifted, partition * n_arms) for partition in range(N_Z)]
    assert np.diff(staggered) == pytest.approx(6.0, abs=1e-3)


def test_a_shorter_train_is_more_inversions_over_the_same_arms():
    whole = design(n_arms=4)
    halved = design(n_arms=4, etl=2)
    counters = halved.evaluate_labels(evolution="adc")
    assert sorted(set(counters["LIN"].tolist())) == list(range(4))
    assert sorted(set(counters["PAR"].tolist())) == list(range(N_Z))
    assert len(set(counters["SEG"].tolist())) == 2 * len(
        set(whole.evaluate_labels(evolution="adc")["SEG"].tolist())
    )


def test_a_train_that_does_not_divide_the_arms_is_refused():
    with pytest.raises(ValueError, match="divide"):
        design(n_arms=4, etl=3)


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
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    assert np.allclose(arm(rows[1]), turn @ arm(rows[0]), atol=1e-4)


def test_every_partition_of_every_arm_is_acquired():
    labels = design(n_arms=3).evaluate_labels(evolution="adc")
    pairs = set(zip(labels["LIN"].tolist(), labels["PAR"].tolist(), strict=False))
    assert pairs == {(arm, partition) for arm in range(3) for partition in range(N_Z)}


def test_the_partitions_are_the_outer_loop():
    """One inversion train covers the arms of a single partition, so the
    partition counter is what changes between trains."""
    labels = design(n_arms=3).evaluate_labels(evolution="adc")
    assert labels["LIN"].tolist()[:6] == [0, 1, 2, 0, 1, 2]
    assert labels["PAR"].tolist()[:6] == [0, 0, 0, 1, 1, 1]


def test_the_first_view_of_a_train_echoes_at_the_inversion_time():
    """TI runs inversion centre to the first spiral's k = 0 crossing, which
    for an outward arm is the first acquired sample."""
    seq = design(n_arms=3)
    k_adc, _, _, _, t_adc = seq.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // (3 * N_Z)
    echo = t_adc[int(np.argmin(np.linalg.norm(k_adc[:, :n_samples], axis=0)))]

    inversions = []
    t = 0.0
    for index in range(1, seq.num_blocks + 1):
        block = seq.get_block(index)
        if block.rf is not None and block.rf.use == "i":
            inversions.append(t + block.rf.delay + block.rf.center)
        t += block.block_duration
    inversions = np.asarray(inversions)

    # The inversion this echo belongs to, not the first of the scan: the
    # steady-state trains invert too.
    assert echo - inversions[inversions < echo].max() == pytest.approx(TI, abs=2e-5)
    assert np.diff(inversions) == pytest.approx(
        np.full(len(inversions) - 1, TR_OUTER), abs=1e-5
    )


def test_the_default_protocol_is_feasible():
    system = pp.Opts()
    report = slot.PLUGIN.validate_protocol(
        system, slot.PLUGIN.get_default_protocol(system)
    )
    assert report["valid"] is True, report["info"]
    assert "arms" in report["info"]
