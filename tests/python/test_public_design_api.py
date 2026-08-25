"""What `pulserver.design` promises about its own surface.

The toolbox is discoverable only if every name on it is a module, is reachable
the documented way, and is written down. None of that survives a refactor on
its own.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver.design import SequenceModule

API_PAGE = Path(__file__).resolve().parents[2] / "docs" / "api" / "python" / "design.md"
# The tree is excluded from the wheel, so it is only ever found in the checkout.
SOURCE_TREE = Path(__file__).resolve().parents[2] / "src" / "python" / "pulserver"


MODULE_CATEGORIES = ("EXCITATION", "PREPARATION", "READOUT", "BASES")


def _module_names() -> list[str]:
    return [name for group in MODULE_CATEGORIES for name in getattr(design, group)]


def test_everything_filed_as_a_module_is_a_sequence_module():
    for name in _module_names():
        exported = getattr(design, name)
        assert inspect.isclass(exported), name
        assert issubclass(exported, SequenceModule), name


def test_the_categories_partition_the_public_surface():
    """Every export is filed under exactly one heading, and nothing else is."""
    categorised = [*_module_names(), *design.PROTOCOL]
    assert sorted(categorised) == sorted(design.__all__)
    assert len(set(categorised)) == len(categorised)


def test_the_protocol_contract_is_reachable_and_is_not_a_module():
    """The contract sits beside the toolbox, and is not part of it."""
    modules = set(_module_names())
    for name in design.PROTOCOL:
        assert hasattr(design, name), name
        assert name not in modules, name


def test_every_shipped_module_can_actually_be_built():
    """Only a base is allowed to be abstract; everything else is a design."""
    for name in _module_names():
        if name in design.BASES:
            continue
        assert inspect.isabstract(getattr(design, name)) is False, name
    # RfModule adds conveniences to the contract without implementing one.
    assert inspect.isabstract(design.RfModule)


def _listed_on_the_page() -> set[str]:
    """The classes the reference page files under its category headings.

    The page sets ``currentmodule`` once and lists bare class names in
    ``autosummary`` blocks, so an entry is an indented identifier on its own
    line -- the same shape whether or not the module prefix is spelled out.
    """
    page = API_PAGE.read_text()
    return {
        name.rpartition(".")[2]
        for name in re.findall(
            r"^\s+((?:pulserver\.design\.)?[A-Za-z]\w+)\s*$", page, re.M
        )
    }


def test_every_public_class_is_on_its_api_reference_page():
    listed = _listed_on_the_page()
    assert set(design.__all__) <= listed, sorted(set(design.__all__) - listed)


def test_the_reference_page_names_nothing_that_is_gone():
    listed = _listed_on_the_page()
    assert listed <= set(design.__all__), sorted(listed - set(design.__all__))


def test_design_holds_modules_and_pypulseq_holds_events():
    """The split is by role, and the two namespaces share no names."""
    assert not set(design.__all__) & set(pp.__all__)


@pytest.mark.parametrize(
    "name",
    [
        "apply_system_derates",
        "ceil_to_raster",
        "round_to_raster",
        "make_rf_spoiling_schedule",
    ],
)
def test_the_helpers_a_plugin_needs_are_in_the_event_layer(name):
    """Raster arithmetic and phase schedules answer from an Opts, not a module."""
    assert name in pp.__all__
    assert not hasattr(design, name)


def test_the_design_tree_holds_only_what_ships():
    """Every module under ``design`` is part of the installed package.

    The toolbox is what a plugin composes from, so a directory here that
    nothing imports is a second answer to a question the package already
    answers -- which is what a reader grepping for a readout would find.
    """
    staged = [
        path
        for path in (SOURCE_TREE / "design").rglob("*")
        if path.is_dir() and path.name.startswith("_disabled")
    ]
    assert not staged, [str(path) for path in staged]
