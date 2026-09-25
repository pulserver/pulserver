"""The repetition the IR segments: the one pypulseqpp finds."""

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


def segmented(sequence):
    """The one subsequence the converter makes of ``sequence``."""
    summary = _ext.summary_from_libraries(
        [conversion_payload(sequence, SYSTEM)], *SCANNER, [0, 1, 2]
    )
    (unit,) = summary["subsequences"]
    return unit


def read(path):
    sequence = pp.Sequence()
    sequence.read(path)
    return sequence


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


def gradients_of_distinct_lengths(count, flat_time):
    """Blocks that repeat neither by definition nor by structure."""

    def build(sequence):
        for index in range(count):
            sequence.add_block(
                pp.make_trapezoid(
                    "x", amplitude=1000, flat_time=flat_time + index * 1e-4
                )
            )

    return build


@pytest.mark.parametrize("name", fixtures())
def test_the_ir_segments_the_repetition_pypulseqpp_finds(name):
    sequence = read(FIXTURES / name)
    assert segmented(sequence)["tr_size"] == sequence.repetition()[0]


def test_a_delay_of_any_length_is_one_definition_so_delays_alone_repeat_every_block(
    tmp_path,
):
    sequence = read(written(tmp_path, "all_delays.seq", alternating_delays))
    # Twelve delays of two lengths are one block played twelve times: how long
    # an interpreter waits at a pure delay is a per-instance value and does
    # not break the period.
    assert segmented(sequence)["tr_size"] == 1


def test_a_sequence_that_plays_something_between_its_delays_repeats_over_all_three(
    tmp_path,
):
    sequence = read(
        written(
            tmp_path, "delays_and_a_gradient.seq", alternating_delays_around_a_gradient
        )
    )
    assert segmented(sequence)["tr_size"] == 3


@pytest.mark.parametrize("tr", [10e-3, 40e-3])
def test_the_repetition_time_counts_each_delay_at_the_duration_it_plays(tmp_path, tr):
    gradient = pp.make_trapezoid("x", flat_area=1000, flat_time=1e-3)
    fill = tr - 1e-3 - pp.calc_duration(gradient)

    def build(sequence):
        for _ in range(6):
            sequence.add_block(pp.make_delay(1e-3))
            sequence.add_block(gradient)
            sequence.add_block(pp.make_delay(fill))

    unit = segmented(read(written(tmp_path, "gradient_in_a_tr.seq", build)))
    assert unit["tr_size"] == 3
    assert unit["tr_duration_us"] == pytest.approx(tr * 1e6)


def test_a_hyper_tr_declared_as_pypulseqpp_writes_it_is_the_repeating_unit(tmp_path):
    def build(sequence):
        alternating_delays_around_a_gradient(sequence)
        sequence.set_definition("TRsize", 6)

    sequence = read(written(tmp_path, "declared_hyper_tr.seq", build))
    assert segmented(sequence)["tr_size"] == 6


def test_a_sequence_that_does_not_repeat_is_one_repetition(tmp_path):
    sequence = read(
        written(tmp_path, "once.seq", gradients_of_distinct_lengths(8, 1e-3))
    )
    assert segmented(sequence)["tr_size"] == len(sequence) == 8


def test_a_sequence_that_does_not_repeat_is_refused_past_15_s(tmp_path):
    sequence = read(
        written(tmp_path, "too_long.seq", gradients_of_distinct_lengths(16, 1.0))
    )
    assert sequence.repetition()[0] == len(sequence)
    assert sequence.duration()[0] > 15.0
    with pytest.raises(ValueError, match="No periodic TR pattern found"):
        segmented(sequence)


def test_a_repetition_that_does_not_start_at_the_first_block_is_refused(
    tmp_path, monkeypatch
):
    sequence = read(
        written(
            tmp_path, "delays_and_a_gradient.seq", alternating_delays_around_a_gradient
        )
    )
    monkeypatch.setattr(sequence, "repetition", lambda: (3, 2))
    with pytest.raises(ValueError, match="repeats from block 2"):
        conversion_payload(sequence, SYSTEM)
