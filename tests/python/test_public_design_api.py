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

import pulserver
import pulserver.design as design
import pulserver.pypulseq as pp
from pulserver import SequenceModule

API_PAGE = Path(__file__).resolve().parents[2] / "docs" / "api" / "design.md"


def test_everything_exported_is_a_sequence_module():
    for name in design.__all__:
        exported = getattr(design, name)
        assert inspect.isclass(exported), name
        assert issubclass(exported, SequenceModule), name


def test_the_categories_partition_the_public_surface():
    """Every export is filed under exactly one heading, and nothing else is."""
    categorised = [*design.EXCITATION, *design.PREPARATION, *design.READOUT, *design.BASES]
    assert sorted(categorised) == sorted(design.__all__)
    assert len(set(categorised)) == len(categorised)


def test_every_shipped_module_can_actually_be_built():
    """Only a base is allowed to be abstract; everything else is a design."""
    for name in design.__all__:
        if name in design.BASES:
            continue
        assert inspect.isabstract(getattr(design, name)) is False, name
    # RfModule adds conveniences to the contract without implementing one.
    assert inspect.isabstract(design.RfModule)


def test_every_public_class_is_on_its_api_reference_page():
    page = API_PAGE.read_text()
    listed = set(re.findall(r"pulserver\.design\.(\w+)", page))
    assert set(design.__all__) <= listed, sorted(set(design.__all__) - listed)


def test_the_reference_page_names_nothing_that_is_gone():
    page = API_PAGE.read_text()
    listed = set(re.findall(r"^\s+pulserver\.design\.(\w+)\s*$", page, flags=re.MULTILINE))
    assert listed <= set(design.__all__), sorted(listed - set(design.__all__))


def test_design_holds_modules_and_pypulseq_holds_events():
    """The split is by role, and the two namespaces share no names."""
    assert not set(design.__all__) & set(pp.__all__)


@pytest.mark.parametrize(
    "name",
    ["apply_system_derates", "ceil_to_raster", "round_to_raster", "make_rf_spoiling_schedule"],
)
def test_the_helpers_a_plugin_needs_are_in_the_event_layer(name):
    """Raster arithmetic and phase schedules answer from an Opts, not a module."""
    assert name in pp.__all__
    assert not hasattr(design, name)


def test_the_disabled_tree_is_not_importable():
    """Kept to port physics from, and deliberately not a package."""
    disabled = Path(pulserver.__file__).parent / "design" / "_disabled"
    assert disabled.is_dir()
    assert not (disabled / "__init__.py").exists()
    with pytest.raises(ModuleNotFoundError):
        __import__("pulserver.design._disabled._readout.epi")
