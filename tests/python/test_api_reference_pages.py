"""What the API reference pages promise about the namespaces they document.

A name that is public but not on a page is invisible, and a name on a page
that is no longer public is a broken link. Neither survives a refactor on its
own, so both are checked here. ``pulserver.design`` has its own page contract
in ``test_public_design_api.py``, and ``pulserver.recon`` in
``recon/test_recon_api.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pulserver.app as app
import pulserver.pypulseq as pp

PAGES = Path(__file__).resolve().parents[2] / "docs" / "api" / "python"

#: An ``autosummary`` block, keyed by the ``toctree`` it generates into: one
#: page carries several namespaces, and only the entries under a given
#: ``toctree`` belong to that one.
_BLOCK = r"autosummary::\n\s+:toctree: \.\./generated/{name}\n(?:\s+:\w+:.*\n)*\n((?:   \S+\n)+)"


def listed(page: str, toctree: str) -> set[str]:
    """The names a page lists under one ``toctree``."""
    text = (PAGES / page).read_text()
    entries: set[str] = set()
    for block in re.findall(_BLOCK.format(name=toctree), text):
        entries |= {line.strip() for line in block.splitlines() if line.strip()}
    return entries


#: The label splits are named in the page's prose rather than given a stub of
#: their own: they are a tuple and a dictionary, so autodoc would render
#: ``tuple``'s and ``dict``'s own documentation beside a page-long value, and
#: what the names mean is ``get_supported_labels``' tables.
_IN_PROSE = {
    "COUNTER_LABELS",
    "ENCODING_COUNTERS",
    "FRAME_COUNTERS",
    "MRD_COUNTERS",
    "FLAG_LABELS",
    "MRD_FLAGS",
    "SCANNER_FLAGS",
    "STICKY_FLAGS",
}


def test_every_pulserver_name_in_the_event_layer_is_documented():
    """``OVERRIDES`` is everything in the namespace that is not upstream's, so
    it is exactly what this project has to document itself."""
    missing = set(pp.OVERRIDES) - listed("pypulseq.md", "pypulseq") - _IN_PROSE
    assert missing == set(), sorted(missing)


def test_the_names_left_to_prose_are_named_there():
    page = (PAGES / "pypulseq.md").read_text()
    for name in _IN_PROSE:
        assert f"`{name}`" in page, name


#: The two sets the page documents. They are reachable on the namespace but
#: deliberately not in ``__all__``: they say which half of a drop-in namespace
#: a name comes from rather than being part of the authoring vocabulary. The
#: finer sets the contract tests are written against stay off the page.
_MEMBERSHIP = {
    "OVERRIDES",
    "UPSTREAM",
}


def test_the_event_layer_page_names_nothing_that_is_gone():
    surplus = listed("pypulseq.md", "pypulseq") - set(pp.__all__) - _MEMBERSHIP
    assert surplus == set(), sorted(surplus)


def test_the_membership_sets_the_page_documents_all_exist():
    for name in _MEMBERSHIP:
        assert isinstance(getattr(pp, name), frozenset), name


def plugins(suffix: str) -> set[str]:
    """The zoo's plugins of one family, off the one flat namespace."""
    return {name for name in app.__all__ if name.endswith(suffix)}


def test_every_zoo_sequence_is_documented():
    """The zoo is the worked-example corpus, and the page is its index."""
    assert listed("apps.md", "app_sequence") == plugins("_sequence")


def test_the_plugin_namespace_is_flat():
    """One access point: a plugin is ``pulserver.app.<name>`` and the families
    are a way of reading the zoo, not a way of importing from it."""
    assert not {"sequence", "recon"} & set(app.__all__)
    assert all(name.endswith(("_sequence", "_recon")) for name in app.__all__), sorted(
        app.__all__
    )
    assert sorted(dir(app)) == sorted(app.__all__)
