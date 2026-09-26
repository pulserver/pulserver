"""Run the examples of the user guide, and hold every public name to an API page."""

import doctest
import importlib
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
GUIDE_PAGES = sorted((DOCS / "user-guide").glob("*.md"))
SUBPACKAGES = ("design", "host", "ir", "mrd", "protocol", "proxy", "recon")


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
    listed = set(
        re.findall(rf"^\| {{obj}}`~pulserver\.{name}\.(\w+)`", page, flags=re.MULTILINE)
    )
    assert set(module.__all__) - listed == set()


GALLERY = DOCS.parent / "gallery"
SCRIPTS = sorted(GALLERY.glob("*/[0-9]*.py"))


@pytest.mark.parametrize(
    "script", SCRIPTS, ids=lambda script: f"{script.parent.name}/{script.name}"
)
def test_every_gallery_script_is_listed_on_a_landing_page(script):
    page = f"/generated/gallery/{script.parent.name}/{script.stem}"
    landing = "\n".join(p.read_text() for p in (DOCS / "examples").glob("*.md"))
    assert f"{{doc}}`{page}`" in landing
    assert f"\n{page}\n" in landing


def test_every_gallery_section_is_built():
    conf = (DOCS / "conf.py").read_text()
    for section in sorted(p for p in GALLERY.iterdir() if p.is_dir()):
        assert f'"../gallery/{section.name}"' in conf
