"""Folding the excitation's slice rephaser into a readout's prewinder.

A slice-selective excitation accrues half its selection moment after the RF
isocentre; the rephaser cancels it. Playing that rephaser as its own block
costs its full duration every TR, so the readouts let it ride the prewinder
they already play. The pairing has two halves and both are load-bearing: the
readout takes ``slice_rephasing``, and the excitation stops playing it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pypulseq")

import pulserver.design as design
import pulserver.pypulseq as pp

OPTS_KW = {"max_grad": 40, "grad_unit": "mT/m", "max_slew": 150, "slew_unit": "T/m/s"}


@pytest.fixture
def system():
    return pp.Opts(**OPTS_KW)


def _slice_moment_after_isocentre(excitation, readout):
    """Net z moment from the RF isocentre to the end of the shot.

    The selection lobe straddles the isocentre, so only half of it dephases
    the slice; everything after it counts in full.
    """
    total = 0.0
    for block_index, block in enumerate(excitation):
        for event in block:
            if getattr(event, "channel", None) != "z" or getattr(event, "type", None) != "trap":
                continue
            total += 0.5 * event.area if block_index == 0 else event.area
    for block in readout:
        for event in block:
            if getattr(event, "channel", None) != "z":
                continue
            kind = getattr(event, "type", None)
            if kind == "trap":
                total += event.area
            elif kind == "grad":
                total += float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))
    return total


def test_without_rephasers_drops_the_block_but_keeps_the_events(system):
    excitation = design.make_slice_selective_pulse(np.deg2rad(15), 5e-3, system=system)
    stripped = excitation.without_rephasers()

    excitation.set_state()
    stripped.set_state()
    assert len(stripped) == len(excitation) - 1
    # Still readable, so `slice_rephasing=...` and `.without_rephasers()` can
    # be written against the same object in either order.
    assert stripped.rephasers == excitation.rephasers
    assert len(excitation) == 2, "the original module must be left untouched"
    assert excitation.without_rephasers().without_rephasers() is not None


def test_without_rephasers_is_a_no_op_when_there_are_none(system):
    hard = design.make_hard_pulse(np.deg2rad(10), duration=1e-3, system=system)
    assert hard.without_rephasers() is hard


@pytest.mark.parametrize("lin_idx", [0, 17, 63])
def test_line_readout_absorbs_the_rephaser_exactly(system, lin_idx):
    excitation = design.make_slice_selective_pulse(np.deg2rad(15), 5e-3, system=system)
    readout = design.make_line_readout(system, (0.22, 0.22), (64, 64), slice_rephasing=excitation.rephasers)

    stripped = excitation.without_rephasers()
    stripped.set_state()
    readout.set_state(lin_idx=lin_idx)
    assert _slice_moment_after_isocentre(stripped, readout) == pytest.approx(0.0, abs=1e-6)


def test_folding_costs_less_than_a_separate_rephasing_block(system):
    excitation = design.make_slice_selective_pulse(np.deg2rad(15), 5e-3, system=system)
    plain = design.make_line_readout(system, (0.22, 0.22), (64, 64))
    folded = design.make_line_readout(system, (0.22, 0.22), (64, 64), slice_rephasing=excitation.rephasers)

    rephaser_block_s = pp.calc_duration(*excitation.rephasers)
    grown_s = folded.duration - plain.duration
    # Some growth is expected -- the prewinder has to widen for the extra
    # area -- but it must be strictly cheaper than paying for a whole block.
    assert 0.0 <= grown_s < rephaser_block_s


def test_epi_readout_absorbs_the_rephaser_exactly(system):
    excitation = design.make_slice_selective_pulse(np.deg2rad(60), 4e-3, system=system)
    mask = np.arange(32)
    readout = design.make_epi_readout(system, (0.22, 0.22), (64, 64), 2, mask, slice_rephasing=excitation.rephasers)

    stripped = excitation.without_rephasers()
    stripped.set_state()
    readout.set_state(lin_idx=0)
    assert _slice_moment_after_isocentre(stripped, readout) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("par_idx", range(8))
def test_a_shared_axis_merges_rather_than_collides(system, par_idx):
    """3D EPI rephases and partition-encodes on the same channel.

    Two gradients cannot occupy one channel in one block, so the rephaser has
    to be summed into the partition encode. What is left over must be exactly
    the kz encoding the unfolded readout would have played.
    """
    excitation = design.make_slice_selective_pulse(np.deg2rad(15), 20e-3, system=system)
    mask = np.column_stack([np.arange(32), np.zeros(32, dtype=int)])
    plain = design.make_epi_readout(system, (0.22, 0.22, 0.12), (64, 64, 8), 2, mask)
    folded = design.make_epi_readout(
        system, (0.22, 0.22, 0.12), (64, 64, 8), 2, mask, slice_rephasing=excitation.rephasers
    )

    stripped = excitation.without_rephasers()
    stripped.set_state()
    plain.set_state(lin_idx=0, par_idx=par_idx)
    folded.set_state(lin_idx=0, par_idx=par_idx)

    kz_encode = _slice_moment_after_isocentre(design.make_hard_pulse(0.1, duration=1e-4, system=system), plain)
    residual = _slice_moment_after_isocentre(stripped, folded) - kz_encode
    assert residual == pytest.approx(0.0, abs=1e-6)

    prewind_channels = [event.channel for event in folded[0] if getattr(event, "channel", None)]
    assert sorted(prewind_channels) == ["x", "y", "z"], "one gradient per channel in the prewind block"


def test_a_rephaser_on_the_readout_axis_is_rejected(system):
    on_readout_axis = pp.make_trapezoid(channel="x", area=100.0, system=system)
    with pytest.raises(ValueError, match="already drives"):
        design.make_line_readout(system, (0.22, 0.22), (64, 64), slice_rephasing=on_readout_axis)
    with pytest.raises(ValueError, match="already drives"):
        design.make_epi_readout(system, (0.22, 0.22), (64, 64), 2, np.arange(32), slice_rephasing=on_readout_axis)


def test_a_single_event_and_a_tuple_are_equivalent(system):
    excitation = design.make_slice_selective_pulse(np.deg2rad(15), 5e-3, system=system)
    one = design.make_line_readout(system, (0.22, 0.22), (64, 64), slice_rephasing=excitation.rephasers[0])
    many = design.make_line_readout(system, (0.22, 0.22), (64, 64), slice_rephasing=excitation.rephasers)
    assert one.t_prephase_s == many.t_prephase_s
    assert one.duration == many.duration


def test_no_rephasing_leaves_the_readout_untouched(system):
    plain = design.make_line_readout(system, (0.22, 0.22), (64, 64))
    explicit = design.make_line_readout(system, (0.22, 0.22), (64, 64), slice_rephasing=None)
    assert explicit.duration == plain.duration
    assert explicit.t_prephase_s == plain.t_prephase_s


@pytest.mark.parametrize("flyback", [False, True])
def test_both_epi_train_families_accept_it(system, flyback):
    """The flyback train is a separate class with its own constructor."""
    excitation = design.make_slice_selective_pulse(np.deg2rad(60), 4e-3, system=system)
    readout = design.make_epi_readout(
        system,
        (0.22, 0.22),
        (64, 64),
        2,
        np.arange(32),
        flyback=flyback,
        slice_rephasing=excitation.rephasers,
    )
    stripped = excitation.without_rephasers()
    stripped.set_state()
    readout.set_state(lin_idx=0)
    assert _slice_moment_after_isocentre(stripped, readout) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "factory, args",
    [
        (design.make_spiral_readout, (0.22, 64, 8)),
        (design.make_radial_readout, (0.22, 64)),
        (design.make_radial_stack_readout, (0.22, 64, 0.12, 8)),
    ],
)
def test_noncartesian_factories_take_the_rephasers_tuple_directly(system, factory, args):
    """`excitation.rephasers` reads the same across every readout family."""
    excitation = design.make_slice_selective_pulse(np.deg2rad(15), 5e-3, system=system)
    one = factory(system, *args, slice_rephasing=excitation.rephasers[0])
    many = factory(system, *args, slice_rephasing=excitation.rephasers)
    assert one.t_prephase_s == many.t_prephase_s
