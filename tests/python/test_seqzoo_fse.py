"""The fast spin-echo modules of the sequence zoo.

What an FSE owes beyond a single spin echo: the CPMG train's k-space must
survive the per-echo sign flips (the real historical bug in this family),
the effective TE must land where the ordering promises, and the 3D module's
orderings and variable flip train must do what their names say.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.seqzoo import fse_2d, fse_3d

BANDWIDTH = 50e3


def design_2d(**kwargs):
    kwargs.setdefault("n_acs", 6)
    kwargs.setdefault("etl", 4)
    kwargs.setdefault("te", 40e-3)
    kwargs.setdefault("tr", 300e-3)
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return fse_2d.main(n_x=32, n_y=16, **kwargs)


def design_3d(**kwargs):
    kwargs.setdefault("n_acs", 4)
    kwargs.setdefault("n_acs_z", 2)
    kwargs.setdefault("etl", 8)
    kwargs.setdefault("te", 60e-3)
    kwargs.setdefault("tr", 250e-3)
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    kwargs.setdefault("readout_crusher_cycles", 1.0)
    return fse_3d.main(n_x=32, n_y=8, n_z=4, slab_thickness=64e-3, **kwargs)


# ----------------------------------------------------------------------
# 2D
# ----------------------------------------------------------------------


def test_the_2d_sequence_is_valid_pulseq():
    is_ok, error_report = design_2d().check_timing()
    assert is_ok, error_report


def test_the_train_survives_its_own_sign_flips():
    """The real k-space round trip: every line lands where its counter says,
    and the read axis stays bounded -- the flyback-under-CPMG failure mode."""
    seq = design_2d()
    labels = seq.evaluate_labels(evolution="adc")
    k_traj_adc, *_ = seq.calculate_kspace()
    n_samples = k_traj_adc.shape[1] // len(labels["LIN"])
    kx = k_traj_adc[0].reshape(-1, n_samples)
    ky = k_traj_adc[1].reshape(-1, n_samples)[:, 0]

    assert float(np.abs(kx).max()) < 32 / (2 * 0.22) + 1.0
    lines = np.rint(ky * 0.22).astype(int) + 8
    assert np.array_equal(lines, labels["LIN"])
    assert sorted(set(labels["LIN"].tolist())) == list(range(16))


def test_the_centre_line_is_acquired_at_the_effective_te():
    seq = design_2d(te=40e-3)
    labels = seq.evaluate_labels(evolution="adc")
    k_traj_adc, _, t_excitation, _, t_adc = seq.calculate_kspace()
    n_samples = k_traj_adc.shape[1] // len(labels["LIN"])
    kx = k_traj_adc[0].reshape(-1, n_samples)
    times = t_adc.reshape(-1, n_samples)

    row = int(np.where(labels["LIN"] == 8)[0][0])
    instant = times[row][int(np.argmin(np.abs(kx[row])))]
    excitation = t_excitation[t_excitation < instant].max()
    assert instant - excitation == pytest.approx(seq.get_definition("TE"), abs=1e-4)


def test_a_te_beyond_the_train_is_clamped_to_its_last_echo():
    kernel = fse_2d.FSE2DKernel(
        pp.Opts(),
        n_x=32,
        n_y=16,
        n_acs=6,
        etl=4,
        te=10.0,
        tr=300e-3,
        readout_bandwidth_hz=BANDWIDTH,
    )
    assert kernel.n_center == 3
    assert kernel.echo_time == pytest.approx(
        float(kernel.repetitions[1][0].echo_times[-1])
    )


def test_the_2d_protocol_reports_the_echo_grid():
    system = pp.Opts()
    protocol = fse_2d.PLUGIN.get_default_protocol(system)
    report = fse_2d.PLUGIN.validate_protocol(system, protocol)
    assert report["valid"] is True, report["info"]
    assert "TEeff" in report["info"] and "ESP" in report["info"]


# ----------------------------------------------------------------------
# 3D: orderings
# ----------------------------------------------------------------------


def _centre_echo(seq):
    labels = seq.evaluate_labels(evolution="adc")
    return next(
        int(echo)
        for line, partition, echo in zip(
            labels["LIN"].tolist(), labels["PAR"].tolist(), labels["ECO"].tolist()
        )
        if (line, partition) == (4, 2)
    )


@pytest.mark.parametrize("ordering", ["linear", "radial", "radial_adaptive", "shuffling"])
def test_every_ordering_covers_the_grid(ordering):
    seq = design_3d(ordering=ordering)
    is_ok, error_report = seq.check_timing()
    assert is_ok, error_report
    labels = seq.evaluate_labels(evolution="adc")
    assert len(set(zip(labels["LIN"].tolist(), labels["PAR"].tolist()))) == 32


def test_the_coherent_orderings_put_the_centre_at_the_target_echo():
    target = round(60e-3 / design_3d().get_definition("EchoSpacing")) - 1
    assert _centre_echo(design_3d(ordering="linear")) == target
    assert _centre_echo(design_3d(ordering="radial_adaptive")) == target
    assert _centre_echo(design_3d(ordering="radial")) == 0


def test_shuffling_is_reproducible_and_scattered():
    """The permutation lives in which view each echo encodes, so that mapping
    -- not the echo stream itself -- is what a seed must pin down."""

    def mapping(seq):
        labels = seq.evaluate_labels(evolution="adc")
        return list(
            zip(labels["LIN"].tolist(), labels["PAR"].tolist(), labels["ECO"].tolist())
        )

    first = mapping(design_3d(ordering="shuffling", shuffle_seed=7))
    again = mapping(design_3d(ordering="shuffling", shuffle_seed=7))
    other = mapping(design_3d(ordering="shuffling", shuffle_seed=8))
    assert first == again
    assert first != other


def test_radial_adaptive_grows_radius_away_from_the_centre_echo():
    """The modulation the ordering exists for: the closer an echo is to the
    target, the closer its views sit to the centre of k-space."""
    seq = design_3d(ordering="radial_adaptive")
    labels = seq.evaluate_labels(evolution="adc")
    target = _centre_echo(seq)
    radii: dict[int, list[float]] = {}
    for line, partition, echo in zip(
        labels["LIN"].tolist(), labels["PAR"].tolist(), labels["ECO"].tolist()
    ):
        radius = ((line - 4) / 8) ** 2 + ((partition - 2) / 4) ** 2
        radii.setdefault(int(echo), []).append(radius)
    means = {echo: np.mean(values) for echo, values in radii.items()}
    distances = sorted(means, key=lambda echo: abs(echo - target))
    ordered_means = [means[echo] for echo in distances]
    assert all(a <= b + 1e-9 for a, b in zip(ordered_means, ordered_means[1:]))


# ----------------------------------------------------------------------
# 3D: variable flips
# ----------------------------------------------------------------------


def test_the_variable_train_modulates_the_played_amplitudes():
    seq = design_3d(variable_flip=True)
    flips = seq.get_definition("RefocusingFlipAngles")
    assert len(flips) == 8
    refocusings = [
        float(np.abs(block.rf.signal).max())
        for index in range(1, seq.num_blocks + 1)
        if (block := seq.get_block(index)).rf is not None and block.rf.use == "r"
    ]
    first_train = refocusings[:8]
    expected = np.asarray(flips) / flips[0]
    assert np.asarray(first_train) / first_train[0] == pytest.approx(
        expected, rel=1e-6
    )


def test_a_constant_train_plays_180s():
    seq = design_3d(variable_flip=False)
    assert set(np.round(seq.get_definition("RefocusingFlipAngles"), 6)) == {180.0}


def test_the_traps_schedule_hits_its_control_points():
    schedule = fse_3d.traps_flip_schedule(
        32, 19, alpha_min_deg=55.0, alpha_center_deg=95.0, alpha_max_deg=150.0
    )
    assert schedule[0] == pytest.approx(150.0)
    assert schedule.min() == pytest.approx(55.0)
    assert schedule[19] == pytest.approx(95.0)
    assert schedule[-1] == pytest.approx(150.0)


def test_the_shuffling_definitions_feed_a_subspace_basis():
    """What the paired reconstruction reads: flips, spacing and train length
    all travel in the file."""
    seq = design_3d(ordering="shuffling")
    assert len(seq.get_definition("RefocusingFlipAngles")) == 8
    assert seq.get_definition("EchoSpacing") > 0
    assert seq.get_definition("EchoTrainLength") == 8
    assert seq.get_definition("ViewOrdering") == "shuffling"
