"""The 3D zero-echo-time module of the sequence zoo.

What ZTE owes: the gradient at full amplitude when the pulse fires, no
return to zero within a shell, constant-step views under per-view
rotations, the missing centre declared, and the offset deferred -- the
scan has no un-rotated readout to bake it into.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.app import zte3D_sequence

N_VIEWS = 16
N_SHOTS = 4
#: Steady-state views at the head of the first shell, played without acquiring.
N_DUMMY = 4


def design(**kwargs):
    kwargs.setdefault("n_views", N_VIEWS)
    kwargs.setdefault("n_shots", N_SHOTS)
    kwargs.setdefault("n_dummy", N_DUMMY)
    kwargs.setdefault("readout_bandwidth_hz", 125e3)
    return zte3D_sequence.main(n_x=32, **kwargs)


def test_the_sequence_is_valid_pulseq():
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


def test_a_view_is_two_blocks_under_one_rotation():
    """Pulse then read, both turned the same way.

    True of a dummy view as well as an acquired one -- the steady-state views
    at the head of the first shell are the same pair without the ADC, since
    what has to settle is the magnetisation and the gradient never stops.
    """
    seq = design()
    rf_blocks = [
        index
        for index in range(1, seq.num_blocks + 1)
        if seq.get_block(index).rf is not None
    ]
    acquired = 0
    for index in rf_blocks:
        pulse, read = seq.get_block(index), seq.get_block(index + 1)
        assert pulse.adc is None
        assert pulse.rotation is not None and read.rotation is not None
        acquired += read.adc is not None
    assert acquired == len(rf_blocks) - N_DUMMY
    assert acquired > 0


def test_the_spokes_run_centre_out_and_cover_the_poles():
    seq = design()
    labels = seq.evaluate_labels(evolution="adc")
    k_adc, *_ = seq.calculate_kspace(dense=False)
    n_samples = k_adc.shape[1] // len(labels["SEG"])
    radius = np.linalg.norm(k_adc.reshape(3, -1, n_samples), axis=0)
    assert np.all(np.diff(radius, axis=1) > 0), "every spoke runs outward"

    ends = k_adc[:, n_samples - 1 :: n_samples]
    directions = (ends / np.linalg.norm(ends, axis=0)).T
    assert directions[:, 2].min() < -0.9 and directions[:, 2].max() > 0.9


def test_the_missing_centre_is_declared():
    seq = design()
    assert seq.get_definition("MissingSamples") >= 1
    assert seq.get_definition("TE") == 0.0


def test_each_shell_carries_its_segment():
    labels = design().evaluate_labels(evolution="adc")
    assert sorted(set(labels["SEG"].tolist())) == list(range(N_SHOTS))
    assert len(labels["SEG"]) == N_SHOTS * N_VIEWS


def test_an_offset_defers_every_adc_and_attaches_the_trajectory():
    seq = design(fov_offset=(5e-3, 0.0, 0.0))
    assert seq._native.has_base_trajectory()
    for index in range(1, seq.num_blocks + 1):
        block = seq.get_block(index)
        if block.adc is not None:
            assert float(block.adc.freq_offset) == 0.0


def test_the_default_protocol_is_feasible():
    system = pp.Opts()
    report = zte3D_sequence.PLUGIN.validate_protocol(
        system, zte3D_sequence.PLUGIN.get_default_protocol(system)
    )
    assert report["valid"] is True, report["info"]
    assert "missing" in report["info"]
