"""The checked-in fixture corpus stays complete, current, and readable.

``tests/python/fixtures/`` is written by ``pulserver.pypulseq`` through the
builders in ``fixture_corpus.py`` (see its docstring for the trust
argument). What is asserted here: every promised file exists, every file
parses, each checked-in file is exactly what today's writer emits for its
builder, the binary twins describe the same sequence as their text
counterparts, and the EPI chains resolve.
"""

from __future__ import annotations

import pytest

import pulserver.pypulseq as pp
from fixture_corpus import BINARY_TWINS, CORPUS, FIXTURES_DIR
from test_pypulseq_roundtrip import assert_same_events

EDGES = ("adc_only", "trap_only", "ext_only")


def load(name: str) -> pp.Sequence:
    seq = pp.Sequence()
    seq.read(FIXTURES_DIR / name)
    return seq


@pytest.mark.parametrize("name", [*CORPUS, *EDGES])
def test_the_fixture_exists_and_parses(name):
    assert (FIXTURES_DIR / f"{name}.seq").exists(), (
        f"{name}.seq missing -- run scripts/regenerate_fixtures.sh"
    )
    assert load(f"{name}.seq").num_blocks > 0


@pytest.mark.parametrize("name", CORPUS)
def test_the_checked_in_fixture_is_what_the_builder_writes_today(name, tmp_path):
    """A drifted writer, builder, or design module shows up as a diff here."""
    fresh = tmp_path / f"{name}.seq"
    CORPUS[name]().write(fresh)
    assert fresh.read_bytes() == (FIXTURES_DIR / f"{name}.seq").read_bytes(), (
        f"{name}.seq no longer matches its builder -- if the change is"
        " intended, run scripts/regenerate_fixtures.sh and review the diff"
    )


@pytest.mark.parametrize("name", BINARY_TWINS)
def test_the_binary_twin_describes_the_same_sequence(name):
    assert_same_events(load(f"{name}.seq"), load(f"{name}.bin"))


@pytest.mark.parametrize("lead", ["epi_2d", "epi_3d"])
def test_the_epi_chain_resolves_inside_the_corpus(lead):
    cursor = FIXTURES_DIR / f"{lead}.seq"
    seen = []
    while cursor is not None:
        assert cursor.exists(), f"chain link missing: {cursor.name}"
        seen.append(cursor.name)
        nxt = load(cursor.name).get_definition("NextSequence")
        cursor = FIXTURES_DIR / nxt if nxt else None
    assert len(seen) >= 2
    assert seen[-1] == f"{lead}_main.seq"
