"""The repeating unit pypulseqpp detects against the one the converter detects."""

from pathlib import Path

import pypulseqpp as pp
import pytest

from pulserver import _ext
from pulserver.ir._source import conversion_payload

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
# The scanner the fixtures convert for; the period does not depend on it.
SCANNER = (42576000.0, 3.0, 2.0, 20.0, 2.0, 20.0)
SYSTEM = pp.Opts(B0=SCANNER[1])


def fixtures():
    return sorted(p.name for p in FIXTURES.glob("*.seq"))


def detected(path):
    """Blocks per repetition, as the converter and as pypulseqpp see it."""
    sequence = pp.Sequence()
    sequence.read(path)
    summary = _ext.summary_from_libraries(
        [conversion_payload(sequence, SYSTEM)], *SCANNER, [0, 1, 2]
    )
    return (
        [part["tr_size"] for part in summary["subsequences"]],
        [sequence._native.repetition()[0]],
    )


def written(tmp_path, name, build):
    sequence = pp.Sequence(pp.Opts())
    build(sequence)
    path = tmp_path / name
    sequence.write(path)
    return path


def alternating_delays(sequence):
    for _ in range(6):
        sequence.add_block(pp.make_delay(1e-3))
        sequence.add_block(pp.make_delay(2e-3))


def alternating_delays_around_a_gradient(sequence):
    gradient = pp.make_trapezoid("x", flat_area=1000, flat_time=1e-3)
    for _ in range(6):
        sequence.add_block(pp.make_delay(1e-3))
        sequence.add_block(gradient)
        sequence.add_block(pp.make_delay(2e-3))


@pytest.mark.parametrize("name", fixtures())
def test_a_fixture_repeats_over_the_same_blocks_either_way(name):
    theirs, ours = detected(FIXTURES / name)
    assert ours == theirs[:1]


def test_a_delay_of_any_length_is_one_definition_so_delays_alone_repeat_every_block(
    tmp_path,
):
    path = written(tmp_path, "all_delays.seq", alternating_delays)
    theirs, ours = detected(path)
    # Twelve delays of two lengths are one block played twelve times: how long
    # an interpreter waits at a pure delay is a per-instance value and does
    # not break the period.
    assert ours == theirs[:1] == [1]


def test_a_sequence_that_plays_something_between_its_delays_repeats_over_all_three(
    tmp_path,
):
    path = written(
        tmp_path, "delays_and_a_gradient.seq", alternating_delays_around_a_gradient
    )
    theirs, ours = detected(path)
    assert ours == theirs[:1] == [3]


@pytest.mark.parametrize("tr", [10e-3, 40e-3])
def test_the_repetition_time_counts_each_delay_at_the_duration_it_plays(tmp_path, tr):
    gradient = pp.make_trapezoid("x", flat_area=1000, flat_time=1e-3)
    fill = tr - 1e-3 - pp.calc_duration(gradient)

    def build(sequence):
        for _ in range(6):
            sequence.add_block(pp.make_delay(1e-3))
            sequence.add_block(gradient)
            sequence.add_block(pp.make_delay(fill))

    sequence = pp.Sequence()
    sequence.read(written(tmp_path, "gradient_in_a_tr.seq", build))
    summary = _ext.summary_from_libraries(
        [conversion_payload(sequence, SYSTEM)], *SCANNER, [0, 1, 2]
    )
    (unit,) = summary["subsequences"]
    assert unit["tr_size"] == 3
    assert unit["tr_duration_us"] == pytest.approx(tr * 1e6)


def test_a_hyper_tr_declared_as_pypulseqpp_writes_it_is_the_repeating_unit(tmp_path):
    def build(sequence):
        alternating_delays_around_a_gradient(sequence)
        sequence.set_definition("TRsize", 6)

    sequence = pp.Sequence()
    sequence.read(written(tmp_path, "declared_hyper_tr.seq", build))
    summary = _ext.summary_from_libraries(
        [conversion_payload(sequence, SYSTEM)], *SCANNER, [0, 1, 2]
    )
    (unit,) = summary["subsequences"]
    assert unit["tr_size"] == 6
