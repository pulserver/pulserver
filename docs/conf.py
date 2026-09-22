"""Configuration for the pulserver documentation."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# The generator lives beside this file rather than on the path the build was
# started from.
sys.path.insert(0, str(Path(__file__).parent))

project = "pulserver"
copyright = "2024-2026, pulserver contributors"  # noqa: A001
author = "pulserver contributors"

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "linkify"]
myst_footnote_transition = False

# Named, not True: with True the pages to read are taken from the environment
# left by the previous build, which is empty on a clean checkout, and no stub is
# written at all. `api_objects.rst` carries every object list and is written
# ahead of autosummary's own handler.
autosummary_generate = ["api_objects.rst"]
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_class_signature = "mixed"
autodoc_preserve_defaults = True

napoleon_numpy_docstring = True
napoleon_use_admonition_for_references = True
# An Attributes section renders as a field list, as Parameters does, rather
# than as one attribute directive per entry.
napoleon_custom_sections = [("Attributes", "params_style")]

pygments_style = "sphinx"
highlight_language = "python"

intersphinx_timeout = 10
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pypulseqpp": ("https://pulserver.github.io/pypulseqpp/latest/", None),
}


class _InventoryOutageFilter(logging.Filter):
    """Drop the warning intersphinx logs when it cannot fetch an inventory.

    The message carries no warning type, so `suppress_warnings` has no name to
    match it by, and the build runs under `-W`: an outage at another project's
    documentation host would fail this build. Cross-references into a project
    whose inventory is missing render as their own text instead.
    """

    _MESSAGE = "failed to reach any of the inventories"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._MESSAGE not in record.getMessage()


def _compact_signature(_app, what, _name, _obj, _options, _signature, return_annotation):
    """Render callable headings compactly; leave data and attributes unchanged."""
    if what in {"function", "method", "class"}:
        return "()", return_annotation
    return None


#: Where the README points its figures for readers on GitHub and PyPI, and
#: what those references become once the same file is the documentation's
#: landing page.
README_ASSETS = (
    "https://raw.githubusercontent.com/pulserver/pulserver/main/docs/_static/",
    "_static/",
)


def _local_readme_assets(_app, docname, source):
    """Use built static assets when the repository README is the index page."""
    if docname == "index":
        source[0] = source[0].replace(*README_ASSETS)


def _included_readme_assets(_app, _relative_path, parent_docname, content):
    """Rewrite the same references in the README pulled in by an ``include``.

    ``source-read`` fires on the landing page before its ``include`` runs, so
    the README's own text is never in the source that handler sees.
    """
    if parent_docname == "index":
        content[0] = content[0].replace(*README_ASSETS)


def _public_bases(_app, _name, _obj, _options, bases):
    """List a private base as the nearest public class it is built on.

    A private class has no page to link to, and a public class built on one
    carries its documentation already.
    """
    bases[:] = [
        next(klass for klass in base.__mro__ if not klass.__name__.startswith("_"))
        if isinstance(base, type)
        else base
        for base in bases
    ]


def _write_api_object_index(app) -> None:
    """Generate every object's stub page from a page outside the navigation tree.

    The API pages list their objects without ``:toctree:``; this collects the
    same lists into an orphan page that writes the stubs, so the sidebar shows
    the API pages without every class, method and attribute stub beneath them.
    """
    from api_objects import write

    write(app.srcdir)


def setup(app):
    """Install the filter ahead of Sphinx's own, which count the warning."""
    app.connect("autodoc-process-bases", _public_bases)
    app.connect("autodoc-process-signature", _compact_signature)
    app.connect("source-read", _local_readme_assets)
    app.connect("include-read", _included_readme_assets)
    # Ahead of autosummary's own handler, which reads the sources for the
    # objects it writes stubs for: a page written after it would only be read
    # on the next build, and its stubs would be a build behind the templates.
    app.connect("builder-inited", _write_api_object_index, priority=100)
    handlers = logging.getLogger("sphinx").handlers
    if not handlers:
        print("conf.py: no Sphinx log handler; an inventory outage will fail the build")
    for handler in handlers:
        handler.filters.insert(0, _InventoryOutageFilter())


PAGES_URL = "https://pulserver.github.io/pulserver"

#: The canonical name of this build: ``latest`` for main, ``stable`` for a
#: release, which is what the page links point at.
DOCS_VERSION = os.environ.get("PULSERVER_DOCS_VERSION", "latest")

#: Which published version this build is: ``latest`` for main, the tag for a
#: release. The switcher marks it and warns on a page older than the newest
#: release.
DOCS_RELEASE = os.environ.get("PULSERVER_DOCS_RELEASE", "latest")

#: The theme decides whether to warn that a page is not the current release by
#: comparing this with the version marked preferred in `versions.json`, not by
#: the `version_match` below -- so leaving it unset warns on every page,
#: `stable` included. A release build carries its tag and matches; `latest` is
#: not a version and does not, which is what the banner is for.
version = release = DOCS_RELEASE

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/pulserver/pulserver",
    "repository_branch": "main",
    # Where the pages live in the repository: the edit button links to the
    # source file under it, and defaults to the repository root without this.
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "home_page_in_toc": True,
    # The list every published version is in, written beside the versions by
    # scripts/publish_docs.py. The page fetches it when it loads, so a build
    # served from anywhere else leaves the switcher out. The theme's check of
    # the list at build time is off: the list exists only once a version has
    # been published.
    "switcher": {
        "json_url": f"{PAGES_URL}/versions.json",
        "version_match": DOCS_RELEASE,
    },
    "check_switcher": False,
    "show_version_warning_banner": True,
    # The sidebar carries the sections and their pages, not every object:
    # individual objects are reached from the tables on the API pages, whose
    # stubs are generated from `docs/api_objects.rst` and so never enter this
    # tree.
    "max_navbar_depth": 3,
    "show_navbar_depth": 1,
}

#: The theme's own sidebar, with the version switcher under the title.
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "version-switcher.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ]
}
html_baseurl = f"{PAGES_URL}/{DOCS_VERSION}/"
html_title = "pulserver documentation"
# The landing page is the repository README, whose figures are rewritten to
# these copies by the handlers above.
html_static_path = ["_static"]
html_css_files = ["pulserver.css"]
html_logo = "_static/pulserver-mark.svg"
