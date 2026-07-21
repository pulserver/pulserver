"""Sphinx configuration for the Pulserver documentation."""

import sys
from pathlib import Path

# Prefer this checkout when another editable pulserver tree is installed in
# the documentation environment.
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if "editable" not in type(finder).__module__
]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

project = "pulserver"
copyright = "2026, INFN-MRI"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "matplotlib.sphinxext.plot_directive",
]

# API prose is the source of truth.  Type annotations remain available to IDEs
# and type checkers, but are intentionally not repeated in rendered signatures
# or parameter descriptions.
autodoc_typehints = "none"
autodoc_class_signature = "separated"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
autosummary_generate = True
autosummary_imported_members = False
templates_path = ["_templates"]

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/INFN-MRI/pulserver",
    "use_repository_button": True,
}

# ``plot`` directives embedded in NumPy-style docstrings execute plotting code
# but only render their figures.  This keeps usage examples concise.
plot_include_source = False
plot_html_show_source_link = False

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
exclude_patterns = ["_build", "README.md"]

# The repository contains reference material that deliberately links to source
# files and headings outside this Sphinx project.  MyST cannot resolve those
# as documentation targets, but they remain valid repository links.
suppress_warnings = ["myst.xref_missing"]
