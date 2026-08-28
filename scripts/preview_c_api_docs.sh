#!/bin/bash
# Render the C and C++ reference and examples on their own, for review.
#
# The full documentation build imports the Python package, executes the plot
# directives and regenerates autosummary stubs and the gallery inside the
# source tree. That is slow, and two of them at once collide over what they
# regenerate. This builds the C pages alone: Doxygen, MyST and Breathe, and
# nothing else.
#
# Usage:
#   bash scripts/preview_c_api_docs.sh            # into docs/_build/c-api
#   bash scripts/preview_c_api_docs.sh --open     # ... and print a file:// URL
#
# Environment:
#   PYTHON_BIN    interpreter carrying sphinx, myst-parser and breathe
#   DOXYGEN_BIN   doxygen to use, if it is not on PATH
#
# The links out of the C section -- to the Python pages, the C++ pages and the
# explanations -- have no targets here and render as plain text. Every link
# within the C section, and every rendered source file, resolves.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPOSITORY/../../../.venv/bin/python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python3"

OUTPUT="$REPOSITORY/docs/_build/c-api"
# The staging tree mirrors the repository's shape, because the pages address
# the sources they render with paths relative to themselves.
ROOT="$REPOSITORY/docs/_build/.c-preview"
STAGING="$ROOT/docs"
WANT_OPEN=false

for argument in "$@"; do
    case "$argument" in
        --open) WANT_OPEN=true ;;
        -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown argument: $argument" >&2; exit 2 ;;
    esac
done

if [ -n "${DOXYGEN_BIN:-}" ]; then
    PATH="$(dirname "$DOXYGEN_BIN"):$PATH"
    export PATH
fi
if ! command -v doxygen >/dev/null 2>&1; then
    echo "doxygen is not on PATH, and the reference is generated from it." >&2
    echo "Install it (apt install doxygen, or conda install -c conda-forge" >&2
    echo "doxygen), or point DOXYGEN_BIN at one." >&2
    exit 1
fi

rm -rf "$ROOT" "$OUTPUT"
mkdir -p "$STAGING/api/c" "$STAGING/api/cpp" "$STAGING/examples/c" \
         "$STAGING/examples/cpp" "$ROOT/examples"
cp "$REPOSITORY"/docs/api/c/*.md "$STAGING/api/c/"
cp "$REPOSITORY"/docs/examples/c/*.md "$STAGING/examples/c/"
cp "$REPOSITORY"/docs/api/cpp/*.md "$STAGING/api/cpp/"
rm -f "$STAGING/api/cpp/pulseq.md" "$STAGING/api/cpp/recon.md"
cp "$REPOSITORY"/docs/examples/cpp/*.md "$STAGING/examples/cpp/"
cp -r "$REPOSITORY/examples/c" "$REPOSITORY/examples/cpp" "$ROOT/examples/"

# The C headers, parsed on their own so every pulseq_* name is unambiguous.
# See docs/_doxygen_xml.py.
(cd "$REPOSITORY/docs" && "$PYTHON_BIN" -c "
import _doxygen_xml, sys
if not _doxygen_xml.run():
    sys.exit('doxygen disappeared between the check and the run')
")

cat > "$STAGING/index.md" <<'EOF'
# C and C++ API and examples

```{toctree}
:maxdepth: 2

api/c/index
examples/c/index
api/cpp/index
examples/cpp/index
```
EOF

cat > "$STAGING/conf.py" <<EOF
"""Sphinx configuration for the standalone C preview."""

project = "pulserver -- C"
extensions = ["myst_parser", "breathe"]
breathe_projects = {
    "pulserver": "$REPOSITORY/docs/_doxygen",
    "pulserver_c": "$REPOSITORY/docs/_doxygen_c",
}
breathe_default_project = "pulserver"
breathe_default_members = ()
breathe_show_include = False
tags.add("doxygen")
myst_enable_extensions = ["dollarmath", "amsmath"]
source_suffix = {".md": "markdown"}
master_doc = "index"
html_theme = "sphinx_book_theme"
html_theme_options = {"show_navbar_depth": 2}
# The Python pages, the C++-only pages and the explanations are out by design.
suppress_warnings = ["myst.xref_missing"]
EOF

# Only these are legitimately absent: the Python pages, the two C++ pages that
# cover the C++-only surface, and the explanations. Anything else is real.
"$PYTHON_BIN" -m sphinx -b html "$STAGING" "$OUTPUT" -q 2>&1 |
    grep -vE "unknown document: '?\.*/?(api/)?(python/|explanations|getting_started|cpp/pulseq|cpp/recon)" || true

echo
echo "C / C++ API and examples: $OUTPUT/index.html"
if $WANT_OPEN; then
    echo "file://$OUTPUT/index.html"
fi
