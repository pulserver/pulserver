"""Run the examples of the user guide, and hold every public name to an API page."""

import doctest
import importlib
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
GUIDE_PAGES = sorted((DOCS / "user-guide").glob("*.md"))
SUBPACKAGES = ("design", "host", "ir", "mrd", "protocol", "recon", "vre")


@pytest.mark.parametrize("page", GUIDE_PAGES, ids=lambda page: page.name)
def test_every_user_guide_example_runs(page):
    results = doctest.testfile(
        str(page), module_relative=False, optionflags=doctest.ELLIPSIS
    )
    assert results.failed == 0


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_every_public_name_is_listed_on_its_api_page(name):
    module = importlib.import_module(f"pulserver.{name}")
    page = (DOCS / "api" / f"{name}.md").read_text()
    listed = set(re.findall(r"^\s{3}(\w+)$", page, flags=re.MULTILINE))
    assert set(module.__all__) - listed == set()
