"""The 3D zero-echo-time module of the sequence zoo.

What ZTE owes: the gradient at full amplitude when the pulse fires, no
return to zero anywhere within a shell, one rotation per shot and none
per view, the missing centre declared, and the offset deferred -- the
scan has no un-rotated readout to bake it into.
"""

from __future__ import annotations

import numpy as np

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

    True of a dummy view as well as an acquired one: a preparation shell is
    an acquiring shell with the ADC left off, so every view in it is the same
    pair of blocks under the same rotation.
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
    # A dummy is a whole shell, so preparation costs N_DUMMY * N_VIEWS
    # non-acquiring views rather than N_DUMMY of them.
    assert acquired == len(rf_blocks) - N_DUMMY * N_VIEWS
    assert acquired > 0


def test_a_shell_is_played_under_a_single_rotation():
    """The shot is the only rotation; the views are waveforms."""
    seq = design()
    turns = [
        np.asarray(seq.get_block(index).rotation, dtype=float).reshape(-1)
        for index in range(1, seq.num_blocks + 1)
        if seq.get_block(index).rotation is not None
    ]
    assert len(turns) == seq.num_blocks
    assert len(np.unique(np.round(turns, 9), axis=0)) == N_SHOTS


def test_the_gradient_never_jumps_between_blocks():
    """Including through the dummies, which hold it rather than turning it."""
    seq = design()
    gradients = seq.waveforms_and_times()[0]
    edge = np.cumsum(
        [seq.get_block(i).block_duration for i in range(1, seq.num_blocks)]
    )
    played = np.stack(
        [
            np.interp(
                np.repeat(edge, 2) + np.tile([-1e-9, 1e-9], len(edge)),
                np.asarray(gradients[axis][0]),
                np.asarray(gradients[axis][1]),
            )
            for axis in range(3)
        ]
    ).reshape(3, -1, 2)
    # A slewing edge moves between the two probes, so what a boundary may not
    # do is move faster than the amplifier can.
    jump = np.linalg.norm(played[:, :, 1] - played[:, :, 0], axis=0)
    assert (jump / 2e-9).max() <= pp.cap_system(pp.Opts(), max_slew=200.0).max_slew


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


def test_a_non_acquiring_dummy_shell_partitions_the_same_as_an_acquiring_one():
    # A dummy is a whole shell with the ADC left off. Whether a block acquires
    # is not part of the structure a repetition is read from, so preparation
    # must not change the detected period -- if it does, the whole scan is
    # read as one TR and every window-based check pays for it.
    from pulserver.app import zte3D_sequence

    sizes = []
    for n_dummy in (0, 2):
        seq = zte3D_sequence.main(
            n_x=32, n_dummy=n_dummy, n_gain_calibration_readouts=0
        )
        sizes.append(seq.tr_size)
    assert sizes[0] == sizes[1]
