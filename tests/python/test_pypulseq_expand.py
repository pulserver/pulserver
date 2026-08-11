"""``Sequence.expand_repeats`` -- the averages, written into the block table.

The arithmetic is covered in ``tests/cpptests/test_pulseq_expand.cpp``, on
sequences built for the purpose. What is checked here is the half that only
exists in Python: that the expansion survives a write and a read back, that
the labels a reader recovers are the ones a reconstruction needs, and that a
real fixture expands to the file an interpreter would have played.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.pypulseq import Sequence

FIXTURES = Path(__file__).resolve().parents[1] / "utils" / "expected"


def load(stem: str) -> Sequence:
    seq = Sequence()
    seq.read(str(FIXTURES / f"{stem}.seq"))
    return seq


def build(sections: list[tuple[float, int | None]]) -> Sequence:
    """A sequence of pure delays, each with an optional ``ONCE`` value.

    The duration is the block's name: it is unique per block and survives
    every operation here, so an assertion on the play order reads as one.
    """
    seq = Sequence(pp.Opts())
    for duration, once in sections:
        events = [pp.make_delay(duration)]
        if once is not None:
            events.append(pp.make_label("ONCE", "SET", once))
        seq.add_block(*events)
    return seq


def labels_at_blocks(seq: Sequence) -> list[dict[str, int]]:
    """Every label in force at each block, replayed the way a reader replays."""
    labelset = seq._native.find_extension_type_id("LABELSET")  # noqa: SLF001
    state: dict[str, int] = {}
    out: list[dict[str, int]] = []
    for index in range(1, seq._native.num_blocks() + 1):  # noqa: SLF001
        link = seq._native.block_events()[index - 1][5]  # noqa: SLF001
        while link != 0:
            type_id, ref, link = seq._native.extension_row(int(link))  # noqa: SLF001
            if labelset and type_id == labelset and ref:
                value, label_id = seq._native.label_set_row(int(ref))  # noqa: SLF001
                state[seq._native.label_name(int(label_id))] = int(value)  # noqa: SLF001
        out.append(dict(state))
    return out


# --------------------------------------------------------------------------
# The play order
# --------------------------------------------------------------------------


def test_the_body_repeats_and_the_ends_do_not():
    seq = build([(1e-3, 1), (2e-3, 0), (3e-3, None), (4e-3, 2)])
    report = seq.expand_repeats(3)

    assert report == {
        "repeats": 3,
        "blocks_before": 4,
        "blocks_after": 8,
        "prep_blocks": 1,
        "body_blocks": 2,
        "cooldown_blocks": 1,
    }
    assert np.allclose(
        seq.block_durations,
        [1e-3, 2e-3, 3e-3, 2e-3, 3e-3, 2e-3, 3e-3, 4e-3],
    )


def test_one_repeat_still_resolves_the_flags():
    """Not a no-op: the flags go, the definition arrives, the order stands."""
    seq = build([(1e-3, 1), (2e-3, 0), (4e-3, 2)])
    report = seq.expand_repeats(1)

    assert report["blocks_after"] == 3
    assert np.allclose(seq.block_durations, [1e-3, 2e-3, 4e-3])
    assert all("ONCE" not in labels for labels in labels_at_blocks(seq))
    assert seq.definitions["IgnoreAverages"] == 1


def test_a_sequence_with_no_flags_repeats_whole():
    seq = build([(1e-3, None), (2e-3, None)])
    seq.expand_repeats(4)
    assert np.allclose(seq.block_durations, [1e-3, 2e-3] * 4)


# --------------------------------------------------------------------------
# The counter
# --------------------------------------------------------------------------


def test_the_average_counter_lands_once_per_repetition():
    """Labels are sticky, so one ``LABELSET`` per average is the whole cost.

    It goes on the first block that repetition actually plays -- which for
    every repetition after the first is the first *body* block, the
    preparation having been skipped.
    """
    seq = build([(1e-3, 1), (2e-3, 0), (3e-3, None), (4e-3, 2)])
    seq.expand_repeats(3)

    avg = [labels.get("AVG", 0) for labels in labels_at_blocks(seq)]
    assert avg == [0, 0, 0, 1, 1, 2, 2, 2]

    # One row per value written -- two, not eight: the first repetition is the
    # sticky default and writes nothing at all.
    avg_id = seq._native.find_label_id("AVG")  # noqa: SLF001
    rows = [
        seq._native.label_set_row(row)  # noqa: SLF001
        for row in range(1, seq._native.num_label_set() + 1)  # noqa: SLF001
    ]
    assert sum(1 for _, label_id in rows if label_id == avg_id) == 2


def test_a_counter_the_design_already_writes_is_refused():
    seq = Sequence(pp.Opts())
    seq.add_block(pp.make_delay(1e-3), pp.make_label("AVG", "SET", 0))
    seq.add_block(pp.make_delay(1e-3), pp.make_label("AVG", "SET", 1))

    with pytest.raises(RuntimeError, match="already writes AVG"):
        seq.expand_repeats(2)


def test_an_empty_label_leaves_the_designs_own_counters_alone():
    seq = Sequence(pp.Opts())
    seq.add_block(pp.make_delay(1e-3), pp.make_label("AVG", "SET", 4))
    seq.expand_repeats(3, label="")

    assert seq._native.num_blocks() == 3  # noqa: SLF001
    assert [labels["AVG"] for labels in labels_at_blocks(seq)] == [4, 4, 4]


# --------------------------------------------------------------------------
# What it refuses
# --------------------------------------------------------------------------


def test_fewer_than_one_repeat_is_refused():
    seq = build([(1e-3, None)])
    with pytest.raises(ValueError, match="at least 1"):
        seq.expand_repeats(0)


def test_a_sequence_that_is_all_preparation_is_refused():
    seq = build([(1e-3, 1), (2e-3, 2)])
    with pytest.raises(RuntimeError, match="no body to repeat"):
        seq.expand_repeats(2)


def test_a_fourth_once_state_is_refused():
    seq = build([(1e-3, 3)])
    with pytest.raises(RuntimeError, match="three-state flag"):
        seq.expand_repeats(2)


# --------------------------------------------------------------------------
# Through a file
# --------------------------------------------------------------------------


def test_the_expansion_survives_a_write_and_a_read(tmp_path):
    seq = build([(1e-3, 1), (2e-3, 0), (4e-3, 2)])
    seq.expand_repeats(3)
    expected = np.asarray(seq.block_durations)

    path = tmp_path / "expanded.seq"
    seq.write(str(path))
    back = Sequence()
    back.read(str(path))

    assert np.allclose(back.block_durations, expected)
    # Read back off the file it is a one-element numeric definition.
    assert np.atleast_1d(back.definitions["IgnoreAverages"])[0] == 1
    assert [labels.get("AVG", 0) for labels in labels_at_blocks(back)] == [0, 0, 1, 2, 2]


def test_a_real_scan_grows_only_its_block_table():
    """The reason this belongs on the design side: repeats are free but for
    the block rows, so a materialised average costs no events at all."""
    seq = load("gre_2d_3sl_3avg")
    blocks = seq._native.num_blocks()  # noqa: SLF001
    before = (
        seq._native.num_gradients(),  # noqa: SLF001
        seq._native.num_rf(),  # noqa: SLF001
        seq._native.num_adc(),  # noqa: SLF001
        seq._native.num_shapes(),  # noqa: SLF001
    )

    report = seq.expand_repeats(2)

    # The fixture carries a preparation and a cooldown of its own, so the
    # table is not simply doubled -- only the body is.
    assert report["prep_blocks"] + report["body_blocks"] + report["cooldown_blocks"] == blocks
    assert report["blocks_after"] == (
        report["prep_blocks"] + 2 * report["body_blocks"] + report["cooldown_blocks"]
    )
    assert (
        seq._native.num_gradients(),  # noqa: SLF001
        seq._native.num_rf(),  # noqa: SLF001
        seq._native.num_adc(),  # noqa: SLF001
        seq._native.num_shapes(),  # noqa: SLF001
    ) == before


def test_the_geometric_counters_can_still_be_derived_afterwards():
    """An expanded scan is still a scan: ``auto_label`` reads it the same way,
    and now has two visits per k-space position to account for."""
    seq = load("gre_2d_3sl_3avg")
    plain, _ = seq.auto_label(skip_apply=True)

    expanded = load("gre_2d_3sl_3avg")
    expanded.expand_repeats(2, label="")
    doubled, _ = expanded.auto_label(skip_apply=True)

    assert np.array_equal(doubled["LIN"], np.concatenate([plain["LIN"]] * 2))
    assert np.array_equal(doubled["SLC"], np.concatenate([plain["SLC"]] * 2))
