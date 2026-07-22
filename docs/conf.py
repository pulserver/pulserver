"""Sphinx configuration for the Pulserver documentation."""

import sys
import warnings
from pathlib import Path

# Prefer this checkout when another editable pulserver tree is installed in
# the documentation environment.
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if "editable" not in type(finder).__module__
]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
# ``_figures`` holds the Bloch simulator and plotting helpers the ``.. plot::``
# directives embedded in docstrings import.  It is documentation-only and is
# deliberately not part of the shipped package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
plot_formats = [("png", 110)]
# Sequence diagrams come from PyPulseq's own plotter, which does not take a
# figure size; give every generated figure room and let the layout engine keep
# the stacked axes legible.  Helpers that set an explicit ``figsize`` win.
plot_apply_rcparams = True
plot_rcparams = {
    "figure.figsize": (8.5, 6.0),
    "figure.autolayout": True,
    "font.size": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
}

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
exclude_patterns = ["_build", "README.md"]

# The repository contains reference material that deliberately links to source
# files and headings outside this Sphinx project.  MyST cannot resolve those
# as documentation targets, but they remain valid repository links.
suppress_warnings = ["myst.xref_missing"]

# Plot directives execute representative designs.  Their defaults are valid;
# these upstream diagnostics concern an ignored optional trapezoid argument,
# a known 1.5.1-versus-1.5.0 preview-reader mismatch, and tight-layout on
# PyPulseq's manually positioned axes.  Keep the documentation build focused
# on actual Sphinx diagnostics while leaving all other warnings visible.
warnings.filterwarnings(
    "ignore",
    message=r"Rise time and fall time is ignored when calculating the shortest duration from `area`\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Using default 4 ms duration for block pulse\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"File version 1\.5\.1 is higher than installed package version 1\.5\.0.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"This figure includes Axes that are not compatible with tight_layout.*",
    category=UserWarning,
)


# ``get_supported_labels`` is re-exported from upstream PyPulseq and opens
# directly on a NumPy section header, so ``autosummary`` finds no first line
# and renders a blank cell on the API landing page.  Supply the missing summary
# at build time; ``autosummary`` reads ``__doc__`` directly, so an
# ``autodoc-process-docstring`` handler would not reach it.  Nothing is written
# back to the upstream package - this lives only in the docs process.
def _patch_upstream_summaries() -> None:
    import pulserver.pypulseq as pp

    existing = pp.get_supported_labels.__doc__ or ""
    if existing.lstrip().startswith("Returns"):
        pp.get_supported_labels.__doc__ = (
            "Return the built-in Pulseq label names, in file order.\n" + existing
        )


_patch_upstream_summaries()
