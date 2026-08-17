"""The balanced SSFP modules of the sequence zoo.

Both dimensionalities in one file: what bSSFP owes beyond a spoiled gradient
echo is the balance (every axis back to k = 0 each repetition), TE pinned at
TR/2, the half-flip catalyst opposite in phase to the first excitation, and
the phase alternation -- so that is what is checked.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.app import bssfp2D_sequence, bssfp3D_sequence

N_X = 32
BANDWIDTH = 25e3


def design_2d(**kwargs):
    kwargs.setdefault("n_acs", 4)
    kwargs.setdefault("n_dummy", 2)
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return bssfp2D_sequence.main(n_x=N_X, n_y=8, **kwargs)


def design_3d(**kwargs):
    kwargs.setdefault("n_acs", 4)
    kwargs.setdefault("n_acs_z", 2)
    kwargs.setdefault("n_dummy", 2)
    kwargs.setdefault("readout_bandwidth_hz", BANDWIDTH)
    return bssfp3D_sequence.main(n_x=N_X, n_y=8, n_z=4, slab_thickness=32e-3, **kwargs)


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_the_sequence_is_valid_pulseq(design):
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_te_is_exactly_half_of_tr(design):
    seq = design()
    assert seq.get_definition("TE") == pytest.approx(seq.get_definition("TR") / 2)


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_every_echo_is_centred(design):
    """k = 0 falls at the middle of every readout.

    An even-sampled readout that straddles the origin has no sample *at* k = 0:
    the two central ones sit at -dk/2 and +dk/2, equidistant. So what is
    asserted is that they straddle it symmetrically -- which is the claim --
    and not which of the two ``argmin`` returns, since that is decided by the
    last bits of an amplitude and not by the sequence.
    """
    seq = design()
    k_traj_adc, *_ = seq.calculate_kspace()
    labels = seq.evaluate_labels(evolution="adc")
    n_samples = k_traj_adc.shape[1] // len(labels["LIN"])
    kx = k_traj_adc[0].reshape(-1, n_samples)

    left = kx[:, n_samples // 2 - 1]
    right = kx[:, n_samples // 2]
    step = np.abs(right - left)

    assert np.all(left < 0.0) and np.all(right > 0.0)
    assert np.allclose(np.abs(left), np.abs(right), rtol=1e-12, atol=0.0)
    assert np.all(np.abs(left) < step)


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_the_catalyst_is_half_the_flip_and_opposite_in_phase(design):
    seq = design()
    pulses = [
        block.rf
        for index in range(1, seq.num_blocks + 1)
        if (block := seq.get_block(index)).rf is not None
    ]
    catalyst, first, second = pulses[0], pulses[1], pulses[2]
    assert np.abs(catalyst.signal).max() == pytest.approx(
        0.5 * np.abs(first.signal).max(), rel=1e-6
    )
    # Alternation makes the first excitation's phase pi; the catalyst opposes
    # it at zero, and successive excitations keep alternating.
    assert float(first.phase_offset) == pytest.approx(np.pi)
    assert float(catalyst.phase_offset) == pytest.approx(0.0)
    assert float(second.phase_offset) == pytest.approx(0.0)


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_the_adc_phase_follows_the_alternation(design):
    seq = design()
    phases = [
        float(block.adc.phase_offset)
        for index in range(1, seq.num_blocks + 1)
        if (block := seq.get_block(index)).adc is not None
    ]
    assert len(set(np.round(phases, 9))) == 2


def test_the_train_is_periodic_at_exactly_one_tr():
    """Successive excitation centres sit one TR apart, catalyst included at
    half -- the periodicity a steady state is. (Balance itself, axis by axis,
    is proven against the waveforms in the design module's own tests.)"""
    seq = design_2d()
    _, _, t_excitation, _, _ = seq.calculate_kspace()
    tr = seq.get_definition("TR")
    gaps = np.diff(t_excitation)
    assert gaps[0] == pytest.approx(tr / 2, abs=1e-9)
    assert gaps[1:] == pytest.approx(np.full(len(gaps) - 1, tr), abs=1e-9)


def test_the_pairs_cover_the_grid_3d():
    seq = design_3d()
    labels = seq.evaluate_labels(evolution="adc")
    assert sorted(set(labels["LIN"].tolist())) == list(range(8))
    assert sorted(set(labels["PAR"].tolist())) == list(range(4))


def test_the_default_protocol_is_feasible():
    system = pp.Opts()
    for module in (bssfp2D_sequence, bssfp3D_sequence):
        report = module.PLUGIN.validate_protocol(
            system, module.PLUGIN.get_default_protocol(system)
        )
        assert report["valid"] is True, report["info"]
        assert "TR/2" in report["info"]
