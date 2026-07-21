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


# Two objects re-exported from upstream PyPulseq have no docstring summary:
# ``SigpyPulseOpts`` has no docstring at all, and ``get_supported_labels``
# opens directly on a NumPy section header.  Either way ``autosummary`` finds
# no first line and renders a blank cell on the API landing page.  Supply the
# missing summary at build time; ``autosummary`` reads ``__doc__`` directly, so
# an ``autodoc-process-docstring`` handler would not reach it.  Nothing is
# written back to the upstream package - this lives only in the docs process.
def _patch_upstream_summaries() -> None:
    import pulserver.pypulseq as pp

    if not (pp.SigpyPulseOpts.__doc__ or "").strip():
        pp.SigpyPulseOpts.__doc__ = "Filter and profile options for a SigPy-designed SLR pulse."

    existing = pp.get_supported_labels.__doc__ or ""
    if existing.lstrip().startswith("Returns"):
        pp.get_supported_labels.__doc__ = (
            "Return the built-in Pulseq label names, in file order.\n" + existing
        )


_patch_upstream_summaries()
