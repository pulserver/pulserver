"""Tests for the optimized continuous-gradient bSSFP/TRUFI readouts."""

from __future__ import annotations

import inspect

import numpy as np
import pulserver.design as design
import pulserver.pypulseq as pp
import pypulseq as upstream
import pytest
from pulserver.design import _readout as readout

OPTS_KW = {"max_grad": 40, "grad_unit": "mT/m", "max_slew": 150, "slew_unit": "T/m/s"}


def _opts():
    return pp.Opts(**OPTS_KW)


def _excitation(system, flip_angle_deg: float, thickness_m: float):
    return design.make_slice_selective_pulse(
        np.deg2rad(flip_angle_deg), thickness_m, duration=0.6e-3, system=system
    )


def _labels(sequence, name: str):
    result = []
    for block_idx in range(1, len(sequence.block_events) + 1):
        block = sequence.get_block(block_idx)
        if block.label is not None:
            result.extend((label.type, label.value) for label in block.label.values() if label.label == name)
    return result


def _as_sequence(train):
    sequence = upstream.Sequence(train.system)
    for block in train:
        sequence.add_block(*block)
    return sequence


def test_2d_bssfp_is_an_rf_owned_continuous_trufi_train() -> None:
    system = _opts()
    excitation = _excitation(system, 35, 5e-3)
    train = design.make_bssfp_readout(
        system, (0.22, 0.22), (64, 8), excitation
    )

    assert isinstance(train, readout.Bssfp2D)
    assert train.num_segments == 1
    assert train.lines_per_segment == 8
    assert train.te_s == pytest.approx(0.5 * train.tr_s)
    assert train._rf.signal == pytest.approx(excitation.rf.signal)
    assert _labels(_as_sequence(train), "ONCE") == [("labelset", 1), ("labelset", 0), ("labelset", 2)]

    adc_phases = [event.phase_offset for block in train for event in block if event.type == "adc"]
    assert adc_phases == pytest.approx([0.0, np.pi] * 4)
    assert _labels(_as_sequence(train), "LIN") == [("labelset", value) for value in range(8)]
    assert _as_sequence(train).check_timing()[0]


def test_bssfp_requires_a_caller_designed_selection_pulse() -> None:
    parameters = inspect.signature(design.make_bssfp_readout).parameters
    assert "excitation" in parameters
    assert {"flip_angle_rad", "slab_thickness_m", "rf_duration_s", "derate"}.isdisjoint(parameters)
    with pytest.raises(TypeError, match="excitation must be"):
        design.make_bssfp_readout(_opts(), (0.22, 0.22), (32, 8), object())


def test_3d_bssfp_segments_cover_kspace_with_identical_shot_structure() -> None:
    system = _opts()
    train = design.make_bssfp_readout(
        system, (0.22, 0.22, 0.12), (32, 4, 4), _excitation(system, 25, 0.12), segment=2
    )
    assert isinstance(train, readout.Bssfp3D)
    assert train.lines_per_segment == 8
    assert train.num_segments == 2

    train.set_state(segment_idx=0)
    segment_zero = _as_sequence(train)
    zero_structure = [tuple(event.type for event in block) for block in train]
    train.set_state(segment_idx=1)
    segment_one = _as_sequence(train)
    one_structure = [tuple(event.type for event in block) for block in train]

    assert len(segment_zero.block_events) == len(segment_one.block_events)
    assert zero_structure == one_structure
    assert segment_zero.check_timing()[0] and segment_one.check_timing()[0]
    assert _labels(segment_zero, "ONCE") == _labels(segment_one, "ONCE") == [
        ("labelset", 1),
        ("labelset", 0),
        ("labelset", 2),
    ]

    def coordinates(sequence):
        return list(
            zip(
                (value for _, value in _labels(sequence, "LIN")),
                (value for _, value in _labels(sequence, "PAR")),
                strict=True,
            )
        )

    assert coordinates(segment_zero) == [(ky, kz) for kz in range(2) for ky in range(4)]
    assert coordinates(segment_one) == [(ky, kz) for kz in range(2, 4) for ky in range(4)]


def test_3d_bssfp_accepts_a_nonselective_caller_designed_pulse() -> None:
    system = _opts()
    excitation = design.make_hard_pulse(np.deg2rad(25), duration=0.5e-3, system=system)
    train = design.make_bssfp_readout(system, (0.22, 0.22, 0.12), (32, 4, 4), excitation, segment=2)

    assert isinstance(train, readout.Bssfp3D)
    assert train._gz_1 is train._gz_2 is None
    sequence = _as_sequence(train)
    assert sequence.check_timing()[0]
    assert _labels(sequence, "ONCE") == [("labelset", 1), ("labelset", 0), ("labelset", 2)]


def test_bssfp_rejects_unequal_3d_segments_and_2d_segmentation() -> None:
    system = _opts()
    with pytest.raises(ValueError, match="divide ny\\*nz"):
        design.make_bssfp_readout(
            system, (0.22, 0.22, 0.12), (32, 4, 3), _excitation(system, 25, 0.12), segment=5
        )
    with pytest.raises(ValueError, match="2D bSSFP"):
        design.make_bssfp_readout(system, (0.22, 0.22), (32, 8), _excitation(system, 25, 5e-3), segment=2)
    with pytest.raises(ValueError, match="2D bSSFP requires"):
        design.make_bssfp_readout(
            system, (0.22, 0.22), (32, 8), design.make_hard_pulse(np.deg2rad(25), duration=0.5e-3, system=system)
        )
