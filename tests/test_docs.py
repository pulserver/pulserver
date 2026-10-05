"""Run the examples of the user guide, hold every public name to an API page, keep the sidebar in the family's order, explanations to their TL;DR and the examples page to its course and Tours."""

import doctest
import importlib
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
GUIDE_PAGES = sorted((DOCS / "user-guide").glob("*.md"))
SUBPACKAGES = (
    "design",
    "host",
    "ir",
    "mrd",
    "protocol",
    "proxy",
    "recon",
    "validate",
    "virtual",
)
#: The six top-level sections, in the order pypulseqpp and bartorch show them.
SECTIONS = (
    "user-guide/index",
    "developer-guide/index",
    "explanations/index",
    "examples/index",
    "api/index",
    "misc/index",
)


def _toctree(page):
    """The entries of the first toctree of a page, in order."""
    block = re.search(r"```\{toctree\}\n(.*?)```", page.read_text(), flags=re.S)
    return [
        line.strip()
        for line in block.group(1).splitlines()
        if line.strip() and not line.startswith(":")
    ]


def test_the_sidebar_lists_the_six_sections_in_order():
    assert tuple(_toctree(DOCS / "index.md")) == SECTIONS


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


EXPLANATIONS = sorted(
    p for p in (DOCS / "explanations").glob("*.md") if p.name != "index.md"
)
TLDR = "```{admonition} TL;DR\n:class: tldr\n"


@pytest.mark.parametrize("page", EXPLANATIONS, ids=lambda page: page.stem)
def test_an_explanation_of_several_sections_opens_with_a_tldr(page):
    text = page.read_text()
    if text.count("\n## ") < 2:
        return
    body = text.split("\n", 1)[1].lstrip("\n")
    assert body.startswith(TLDR), f"{page.name} does not open with a TL;DR"


def test_no_landing_page_carries_a_tldr():
    for page in DOCS.rglob("index.md"):
        if "generated" not in page.parts and "build" not in page.parts:
            assert "TL;DR" not in page.read_text(), page


def _section(text, heading):
    return text.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


@pytest.mark.parametrize(
    ("heading", "directory"), [("Course", "01-course"), ("Tours", "02-tours")]
)
def test_the_examples_page_lists_the_course_then_the_tours_in_order(heading, directory):
    text = (DOCS / "examples" / "index.md").read_text()
    assert text.index("\n## Course\n") < text.index("\n## Tours\n")
    table = _section(text, heading)
    pages = [
        f"/generated/gallery/{directory}/{script.stem}"
        for script in sorted((GALLERY / directory).glob("[0-9]*.py"))
    ]
    assert pages
    positions = [table.index(f"{{doc}}`{page}`") for page in pages]
    assert positions == sorted(positions)
