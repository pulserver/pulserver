"""The checked-in fixture corpus stays complete, current, and readable.

``tests/python/fixtures/`` is written by ``pulserver.pypulseq`` through the
builders in ``fixture_corpus.py`` (see its docstring for the trust
argument). What is asserted here: every promised file exists, every file
parses, each checked-in file is what today's writer emits for its builder,
every builder's gradients stay inside the limits it declares, the
binary twins describe the same sequence as their text counterparts, and the
EPI chains resolve.
"""

from __future__ import annotations

from math import isclose

import pytest

import pulserver.pypulseq as pp
from fixture_corpus import BINARY_TWINS, CORPUS, FIXTURES_DIR
from test_pypulseq_roundtrip import assert_same_events

EDGES = ("adc_only", "trap_only", "ext_only")

# A shape sample is written with nine significant digits, and the last of them
# follows the host's libm: the same builder on two machines can differ there by
# a few units in the last place. Everything else -- section headers, counts,
# ids, timings on the raster -- is compared as written.
SAMPLE_TOLERANCE = 1e-7


def body(text: str) -> list[str]:
    """The lines before ``[SIGNATURE]``, whose hash covers every digit of them."""
    lines = text.splitlines()
    return lines[: lines.index("[SIGNATURE]")] if "[SIGNATURE]" in lines else lines


def agree(written: str, stored: str) -> bool:
    """Whether two tokens are the same, allowing a shape sample its last digit."""
    if written == stored:
        return True
    try:
        return isclose(
            float(written), float(stored), rel_tol=SAMPLE_TOLERANCE, abs_tol=1e-12
        )
    except ValueError:
        return False


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
def test_a_builder_stays_inside_the_limits_it_declares(name):
    """No family designs a gradient its own ``system`` would refuse.

    A plugin hands one system object to every module it builds and then to
    the ``Sequence``, so anything that lowers those limits partway through
    leaves the events built before it over the ceiling the file goes on to
    declare. The peak is judged in the stored frame, which is the frame the
    declared limits are stated in.
    """
    is_ok, message = CORPUS[name]().check_hardware_limits()
    assert is_ok, f"{name}: {message}"


@pytest.mark.parametrize("name", CORPUS)
def test_the_checked_in_fixture_is_what_the_builder_writes_today(name, tmp_path):
    """A drifted writer, builder, or design module shows up as a diff here."""
    path = tmp_path / f"{name}.seq"
    CORPUS[name]().write(path)
    written = path.read_text()
    stored = (FIXTURES_DIR / f"{name}.seq").read_text()
    drift = (
        f"{name}.seq no longer matches its builder -- if the change is"
        " intended, run scripts/regenerate_fixtures.sh and review the diff"
    )
    assert ("[SIGNATURE]" in written) == ("[SIGNATURE]" in stored), drift

    written_lines, stored_lines = body(written), body(stored)
    assert len(written_lines) == len(stored_lines), drift
    for number, (mine, theirs) in enumerate(
        zip(written_lines, stored_lines, strict=True), 1
    ):
        if mine == theirs:
            continue
        detail = f"{drift}\nline {number}:\n  written: {mine}\n  stored:  {theirs}"
        mine_tokens, their_tokens = mine.split(), theirs.split()
        assert len(mine_tokens) == len(their_tokens), detail
        for written_token, stored_token in zip(mine_tokens, their_tokens, strict=True):
            assert agree(written_token, stored_token), detail


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
