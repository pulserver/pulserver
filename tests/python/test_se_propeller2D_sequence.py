"""The 2D PROPELLER spin echo of the sequence zoo.

What the slot owes: a true spin echo per blade (excitation, one 180, the
blade's centre line at TE with the 180 at its midpoint), every blade
sampling the centre of k-space, and one blade waveform serving every angle.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.app import se_propeller2D_sequence

N_X = 32
WIDTH = 4
BLADES = 4


def design(**kwargs):
    kwargs.setdefault("blade_width", WIDTH)
    kwargs.setdefault("n_blades", BLADES)
    kwargs.setdefault("te", 60e-3)
    kwargs.setdefault("tr", 200e-3)
    kwargs.setdefault("readout_bandwidth_hz", 125e3)
    return se_propeller2D_sequence.main(n_x=N_X, **kwargs)


def test_the_sequence_is_valid_pulseq():
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


def test_each_blade_is_one_spin_echo_train():
    """A blade is sampled Cartesian, so its lines are phase encodes: LIN counts
    within the blade and SEG says which blade."""
    labels = design().evaluate_labels(evolution="adc")
    assert sorted(set(labels["SEG"].tolist())) == list(range(BLADES))
    assert sorted(set(labels["LIN"].tolist())) == list(range(WIDTH))
    assert len(labels["LIN"]) == BLADES * WIDTH


def test_the_blades_centre_line_meets_the_spin_echo():
    seq = design(te=60e-3)
    labels = seq.evaluate_labels(evolution="adc")
    k_adc, _, t_excitation, t_refocusing, t_adc = seq.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // len(labels["LIN"])

    row = next(
        index
        for index, (blade, line) in enumerate(
            zip(labels["SEG"].tolist(), labels["LIN"].tolist(), strict=False)
        )
        if blade == 0 and line == WIDTH // 2
    )
    kx = k_adc[0, row * n_samples : (row + 1) * n_samples]
    instant = t_adc[row * n_samples + int(np.argmin(np.abs(kx)))]
    excitation = t_excitation[t_excitation < instant].max()
    refocusing = t_refocusing[(t_refocusing > excitation) & (t_refocusing < instant)]

    assert instant - excitation == pytest.approx(60e-3, abs=2e-4)
    assert refocusing[0] - excitation == pytest.approx(30e-3, abs=2e-5)


def test_every_blade_samples_the_centre_of_k_space():
    seq = design()
    labels = seq.evaluate_labels(evolution="adc")
    k_adc, *_ = seq.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // len(labels["LIN"])
    delta_k = 1.0 / 0.22
    for blade in range(BLADES):
        rows = [
            index
            for index, value in enumerate(labels["SEG"].tolist())
            if value == blade
        ]
        radius = min(
            float(
                np.hypot(
                    k_adc[0, row * n_samples : (row + 1) * n_samples],
                    k_adc[1, row * n_samples : (row + 1) * n_samples],
                ).min()
            )
            for row in rows
        )
        assert radius < delta_k


def test_one_blade_waveform_serves_every_angle():
    few = design(n_blades=2)
    many = design(n_blades=6)
    assert many._native.num_shapes() == few._native.num_shapes()


def test_an_offset_defers_the_rotated_adc():
    seq = design(fov_offset=(5e-3, 0.0, 0.0))
    assert seq._native.has_base_trajectory()


def test_the_default_protocol_is_feasible():
    system = pp.Opts()
    report = se_propeller2D_sequence.PLUGIN.validate_protocol(
        system, se_propeller2D_sequence.PLUGIN.get_default_protocol(system)
    )
    assert report["valid"] is True, report["info"]
    assert "blades" in report["info"]
