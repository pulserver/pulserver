"""The rule by which a repetition's blocks become the units a machine plays."""

import shutil
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver.ir import Grouping, convert, repetition_gradients, summary

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SEQUENCES = (
    "gre_2d_3sl.seq",
    "epi_2d_main.seq",
    "mprage_stack_of_spirals_3d.seq",
    "zte_3d.seq",
)


def _copied(name, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    shutil.copy(FIXTURES / name, path)
    return path


def _grouped(path, grouping):
    """Return the units a grouping makes, and the gradient they play over time."""
    convert(path, SYSTEM, grouping=grouping, cache_ext=".pseg")
    return (
        summary(path, SYSTEM, cache_ext=".pseg")["num_segments"],
        repetition_gradients(path, cache_ext=".pseg"),
    )


@pytest.mark.parametrize("name", SEQUENCES)
def test_the_default_grouping_is_what_a_conversion_does_unasked(name, tmp_path):
    """Stating the rule a machine already wanted changes nothing it is given."""
    stated = _copied(name, tmp_path)
    unasked = _copied(name, tmp_path / "unasked")
    convert(stated, SYSTEM, grouping=Grouping(), cache_ext=".pseg")
    convert(unasked, SYSTEM, cache_ext=".pseg")
    assert (
        stated.with_suffix(".pseg").read_bytes()
        == unasked.with_suffix(".pseg").read_bytes()
    )


def test_keeping_a_units_edge_delays_leaves_fewer_units(tmp_path):
    """A machine that pays switching time per unit wants as few as it can have."""
    path = _copied("gre_2d_3sl.seq", tmp_path)
    split, _ = _grouped(path, Grouping())
    whole, _ = _grouped(path, Grouping(split_edge_delays=False))
    assert whole < split


def test_a_machine_that_may_begin_a_unit_under_a_gradient_gets_more_units(tmp_path):
    """Where a boundary may fall is the machine's to state, not this library's.

    A radial shell rests its gradients once a shot, so a boundary only where
    they rest leaves the shot one unit; a machine that can set a gradient on
    entry has its pick of the positions between.
    """
    path = _copied("zte_3d.seq", tmp_path)
    resting, _ = _grouped(path, Grouping())
    anywhere, _ = _grouped(path, Grouping(boundary_gradient_hz_per_m=1e9))
    assert anywhere > resting


@pytest.mark.parametrize("name", SEQUENCES)
@pytest.mark.parametrize(
    "grouping",
    [
        Grouping(split_edge_delays=False),
        Grouping(boundary_gradient_hz_per_m=1e9),
        Grouping(split_by_pulses=False, split_by_readouts=False),
    ],
    ids=["edge-delays-kept", "boundary-anywhere", "unrefined"],
)
def test_a_grouping_moves_the_boundaries_and_not_the_gradient(name, grouping, tmp_path):
    """Grouping is where a unit begins; the scan it plays is the same scan.

    Checked on the repetition of most gradient energy, which is the one a
    scanner costs its gradient heating and its acoustics from.
    """
    path = _copied(name, tmp_path)
    _, resting = _grouped(path, Grouping())
    _, regrouped = _grouped(path, grouping)
    assert regrouped["duration_us"] == resting["duration_us"]
    assert regrouped["energy"] == pytest.approx(resting["energy"], rel=1e-9)
    for axis in ("time_us", "gx", "gy", "gz"):
        if axis in resting and axis in regrouped:
            assert np.asarray(regrouped[axis]) == pytest.approx(
                np.asarray(resting[axis]), rel=1e-6, abs=1e-6
            )
