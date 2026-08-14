"""The non-Cartesian gradient echoes of the sequence zoo.

What the family owes: rotations as extensions (one waveform however many
shots), coverage of the declared views, and the server-mode FOV deferral
these slots exist to demonstrate.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.seqzoo import (
    gre_radial_2d,
    gre_spiral_2d,
    gre_stack_of_spirals_3d,
    gre_stack_of_stars_3d,
)

BANDWIDTH = 125e3


def radial(**kwargs):
    kwargs.setdefault("n_spokes", 8)
    kwargs.setdefault("n_dummy", 2)
    kwargs.setdefault("tr", 20e-3)
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return gre_radial_2d.main(n_x=32, **kwargs)


def spiral(**kwargs):
    kwargs.setdefault("n_arms", 4)
    kwargs.setdefault("n_dummy", 2)
    kwargs.setdefault("tr", 20e-3)
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return gre_spiral_2d.main(n_x=32, **kwargs)


def stars(**kwargs):
    kwargs.setdefault("n_spokes", 4)
    kwargs.setdefault("n_dummy", 2)
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return gre_stack_of_stars_3d.main(n_x=32, n_z=4, slab_thickness=64e-3, **kwargs)


def spirals(**kwargs):
    kwargs.setdefault("n_arms", 4)
    kwargs.setdefault("n_dummy", 2)
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return gre_stack_of_spirals_3d.main(n_x=32, n_z=4, slab_thickness=64e-3, **kwargs)


DESIGNS = {"radial": radial, "spiral": spiral, "stars": stars, "spirals": spirals}


@pytest.mark.parametrize("name", list(DESIGNS))
def test_the_sequence_is_valid_pulseq(name):
    is_ok, error_report = DESIGNS[name]().check_timing()
    assert is_ok, error_report


@pytest.mark.parametrize("name", list(DESIGNS))
def test_every_acquisition_is_rotated(name):
    seq = DESIGNS[name]()
    for index in range(1, seq.num_blocks + 1):
        block = seq.get_block(index)
        if block.adc is not None:
            assert block.rotation is not None


def test_one_spoke_waveform_serves_every_angle():
    """The rotation extension's whole point: shapes do not grow with spokes."""
    few = radial(n_spokes=4)
    many = radial(n_spokes=16)
    assert many._native.num_shapes() == few._native.num_shapes()


def test_the_spokes_turn_by_the_golden_angle():
    seq = radial(n_spokes=8)
    n_adc = sum(
        1 for i in range(1, seq.num_blocks + 1) if seq.get_block(i).adc is not None
    )
    k_adc = seq.calculate_kspace(dense=False)[0]
    n_samples = k_adc.shape[1] // n_adc
    first = k_adc[:2, :n_samples]
    angle = float(gre_radial_2d.spoke_angles(8, "golden")[1])
    turn = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    assert np.allclose(k_adc[:2, n_samples : 2 * n_samples], turn @ first, atol=1e-6)


def test_a_spiral_arm_starts_at_the_centre_and_reaches_kmax():
    seq = spiral(n_arms=4)
    n_adc = sum(
        1 for i in range(1, seq.num_blocks + 1) if seq.get_block(i).adc is not None
    )
    k_adc = seq.calculate_kspace(dense=False)[0]
    n_samples = k_adc.shape[1] // n_adc
    radius = np.hypot(k_adc[0, :n_samples], k_adc[1, :n_samples])
    assert radius[0] < 3.0
    assert radius.max() == pytest.approx(32 / (2 * 0.22), rel=0.05)


@pytest.mark.parametrize("name", ["stars", "spirals"])
def test_a_stack_covers_every_partition(name):
    labels = DESIGNS[name]().evaluate_labels(evolution="adc")
    assert sorted(set(labels["PAR"].tolist())) == [0, 1, 2, 3]


@pytest.mark.parametrize("name", list(DESIGNS))
def test_an_offset_defers_the_rotated_adc_and_attaches_the_trajectory(name):
    """The server-mode deferral this family demonstrates: the ADC is left to
    the consumer, which gets the base trajectory to finish it with."""
    seq = DESIGNS[name](fov_offset=(5e-3, 0.0, 0.0))
    assert seq._native.has_base_trajectory()
    first_adc = next(
        seq.get_block(i).adc
        for i in range(1, seq.num_blocks + 1)
        if seq.get_block(i).adc is not None
    )
    assert float(first_adc.freq_offset) == 0.0


@pytest.mark.parametrize(
    "module",
    [gre_radial_2d, gre_spiral_2d, gre_stack_of_stars_3d, gre_stack_of_spirals_3d],
)
def test_the_default_protocol_is_feasible(module):
    system = pp.Opts()
    report = module.PLUGIN.validate_protocol(
        system, module.PLUGIN.get_default_protocol(system)
    )
    assert report["valid"] is True, report["info"]
