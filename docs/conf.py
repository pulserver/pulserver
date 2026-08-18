"""Sphinx configuration for the Pulserver documentation."""

import re
import site
import sys
import warnings
from pathlib import Path

# ``_figures`` holds the Bloch simulator and plotting helpers the ``.. plot::``
# directives embedded in docstrings import.  It is documentation-only and is
# deliberately not part of the shipped package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

_REPOSITORY = Path(__file__).resolve().parents[1]


def _documented_elsewhere(imported: Path) -> bool:
    """True when ``imported`` is neither this checkout nor this environment."""
    if _REPOSITORY in imported.parents:
        return False
    roots = [*site.getsitepackages(), site.getusersitepackages()]
    return not any(Path(root).resolve() in imported.parents for root in roots)


def _resolve_pulserver() -> None:
    """Check which Pulserver is being documented, and map its plugins.

    The package is imported rather than read off ``src/python`` directly,
    because the compiled extension exists only once it has been built: a
    plain install supplies both from site-packages, an editable install of
    this checkout supplies the sources from here and the extension from
    there.  An editable install pointing at a *different* checkout would
    silently document that one, which is what this refuses.

    The plugin mapping is the second half: the wheel maps ``examples/`` onto
    ``pulserver.app.sequence`` and ``pulserver.app.recon``, and the plugin
    objects carry those dotted names -- which is what ``autosummary``
    follows.
    """
    import pulserver
    import pulserver.app

    imported = Path(pulserver.__file__).resolve()
    if _documented_elsewhere(imported):
        raise RuntimeError(
            f"the documentation builds from {_REPOSITORY}, but `import "
            f"pulserver` resolved to {imported}, which belongs to neither "
            f"this checkout nor this environment. Install this checkout "
            f"(`pip install -e .`) before building the documentation."
        )

    examples = str(_REPOSITORY / "examples")
    if examples not in pulserver.app.__path__:
        pulserver.app.__path__.append(examples)


_resolve_pulserver()

project = "pulserver"
copyright = "2026, the Pulserver authors"

# The version the rendered pages report. Read from the installed
# distribution, so a tagged build says the tag and a working checkout says
# whatever is installed, rather than a number kept in step by hand.
try:
    from importlib.metadata import version as _distribution_version

    release = _distribution_version("pulserver")
except Exception:  # pragma: no cover - an uninstalled source checkout
    release = "0.0.0.dev0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
]

# The explanation pages carry real formulae. Without ``dollarmath`` MyST does
# not treat ``$...$`` / ``$$...$$`` as maths at all and renders the LaTeX
# verbatim, dollar signs and all. ``amsmath`` adds the numbered environments
# (``align``, ``gather``) alongside it.
myst_enable_extensions = ["dollarmath", "amsmath"]

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
    "repository_url": "https://github.com/pulserver/pulserver",
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
#
# ``ref.ref`` covers labels that only exist in a dependency's own
# documentation: autodoc pulls third-party docstrings into the API pages, and
# those docstrings cross-reference galleries that are not built here.
suppress_warnings = ["myst.xref_missing", "ref.ref"]

# Autodoc renders docstrings from dependencies alongside ours -- a re-exported
# class brings its own documentation with it -- and those were written for
# their own Sphinx configuration.  Supply the substitutions they expect.
rst_prolog = """
.. |sep| replace:: /
"""

#: ``sphinxcontrib-bibtex`` citation roles, which some dependencies use in
#: their docstrings.  This project has no bibliography, so the roles are
#: rewritten to the key they cite before the docstring is parsed; the
#: alternative is a bibliography extension that would then have nothing to
#: resolve the keys against.
_CITATION_ROLE = re.compile(r":(?:foot)?cite:(?:[tp]:)?`([^`]*)`")

#: A field name written with a space before its closing colon
#: (``:param int x :``).  docutils does not read that as a field, so the list
#: ends early and the rest of the docstring lands as a parse error.
_LOOSE_FIELD_NAME = re.compile(
    r"^(\s*):(param|type|returns?|rtype|raises?|ivar|vartype)([^:]*?)\s+:"
)


def _repair_third_party_docstrings(app, what, name, obj, options, lines):
    """Make dependencies' docstrings parseable, in place.

    Autodoc renders re-exported classes with the docstrings they came with,
    and those are written for their own Sphinx setup.  Nothing here is
    written back to the dependency: it only affects what this build parses.
    """
    for index, line in enumerate(lines):
        if "cite:" in line:
            line = _CITATION_ROLE.sub(r"\1", line)
        line = _LOOSE_FIELD_NAME.sub(r"\1:\2\3:", line)
        lines[index] = line

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


def setup(app):
    """Sphinx entry point."""
    app.connect("autodoc-process-docstring", _repair_third_party_docstrings)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
