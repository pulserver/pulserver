"""What the C and C++ reference pages promise about the headers.

The pages name entities one at a time rather than pulling in whole files, so a
header that grows a function grows no documentation on its own. This is the
check that says so, and the other direction -- a page naming something the
headers no longer declare, which renders as a build warning nobody reads.

Doxygen is a system package, so this lane skips itself where it is missing,
the way the native test lanes do.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
PAGES = _ROOT / "docs" / "api"

sys.path.insert(0, str(_ROOT / "docs"))
_doxygen_xml = pytest.importorskip("_doxygen_xml")

#: Nested names come back from Doxygen in their own right, and are documented
#: by the ``:members:`` of the class that holds them rather than named again.
_NESTED = re.compile(r"::\w+::")

#: Doxygen reports the C++ ``pulseq`` sources twice, once at each precision.
#: ``raw64`` is the same declarations, and the page says so in prose.
_SECOND_INSTANTIATION = "pulseq::raw64"


def documented() -> set[str]:
    """Every entity the C and C++ pages name."""
    named: set[str] = set()
    for page in PAGES.glob("c*/*.md"):
        for match in re.finditer(r"```\{doxygen\w+\} ([^\n(]+)", page.read_text()):
            named.add(match.group(1).strip())
    return named


@pytest.fixture(scope="module")
def declared() -> set[str]:
    """Every entity Doxygen finds in the public headers."""
    if not _doxygen_xml.run():
        pytest.skip("doxygen is not installed")
    found = _doxygen_xml.entities()
    return {
        name
        for name in found
        if not _NESTED.search(name) and not name.startswith(_SECOND_INSTANTIATION)
    }


def test_every_declared_entity_is_named_on_a_page(declared):
    """A public header that grows an entity grows a line on its page."""
    missing = declared - documented()
    assert missing == set(), sorted(missing)


def test_every_named_entity_is_still_declared(declared):
    """An overload is named with its arguments, so it is checked by its stem."""
    named = {name.split("(")[0].strip() for name in documented()}
    # ``doxygenfile`` takes a header name rather than an entity.
    stale = {name for name in named if not name.endswith((".h", ".hpp"))} - declared
    assert stale == set(), sorted(stale)
