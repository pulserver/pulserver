"""Sphinx configuration for the Pulserver documentation."""

import sys
from importlib.util import find_spec
from pathlib import Path

# Prefer this checkout when another editable pulserver tree is installed in
# the documentation environment.
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if not type(finder).__module__.startswith(("_editable", "_pge_editable"))
]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

project = "pulserver"
copyright = "2026, INFN-MRI"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "furo" if find_spec("furo") else "alabaster"
exclude_patterns = ["_build"]
