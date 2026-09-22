"""Sphinx configuration for the pulserver documentation."""

from __future__ import annotations

import logging
import os

project = "pulserver"
copyright = "2024-2026, Matteo Cencini"  # noqa: A001
author = "Matteo Cencini"

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
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "linkify"]
myst_footnote_transition = False

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_preserve_defaults = True

napoleon_numpy_docstring = True
napoleon_custom_sections = [("Attributes", "params_style")]

intersphinx_timeout = 10
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pypulseqpp": ("https://pulserver.github.io/pypulseqpp/latest/", None),
}


class _InventoryOutageFilter(logging.Filter):
    """Drop the untyped warning intersphinx logs when an inventory is unreachable.

    The build runs under ``-W`` and ``suppress_warnings`` cannot name this
    message, so an outage at another project's documentation host would fail it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "failed to reach any of the inventories" not in record.getMessage()


def setup(app):
    for handler in logging.getLogger("sphinx").handlers:
        handler.filters.insert(0, _InventoryOutageFilter())


PAGES_URL = "https://pulserver.github.io/pulserver"

#: The canonical name of this build: ``latest`` for main, ``stable`` for a
#: release, which is what the page links point at.
DOCS_VERSION = os.environ.get("PULSERVER_DOCS_VERSION", "latest")

#: Which published version this build is: ``latest`` for main, the tag for a
#: release. The switcher marks it, and the theme warns on a page whose version
#: is not the one ``versions.json`` marks preferred.
DOCS_RELEASE = os.environ.get("PULSERVER_DOCS_RELEASE", "latest")
version = release = DOCS_RELEASE

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/pulserver/pulserver",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "home_page_in_toc": True,
    # The list every published version is in, written beside the versions by
    # scripts/publish_docs.py and fetched when a page loads. It exists only
    # once a version has been published, so the build does not check it.
    "switcher": {
        "json_url": f"{PAGES_URL}/versions.json",
        "version_match": DOCS_RELEASE,
    },
    "check_switcher": False,
    "show_version_warning_banner": True,
    # Sections and their pages; the per-object pages autosummary writes are
    # reached from the tables on the API pages, not from the sidebar.
    "max_navbar_depth": 2,
    "show_navbar_depth": 1,
}
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
