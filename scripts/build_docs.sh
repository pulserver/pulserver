#!/usr/bin/env bash
# Build the HTML documentation into docs/build/html. Any Sphinx warning fails
# the build; extra arguments go to sphinx-build.
#
# autodoc imports the installed pulserver, so install this checkout first:
# pip install -e '.[doc]'.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! "$PYTHON_BIN" -c "import sphinx, myst_parser, sphinx_book_theme, sphinx_copybutton, linkify_it" 2>/dev/null; then
    echo "build_docs.sh: the documentation tools are missing; install them with pip install '.[doc]'" >&2
    exit 1
fi

# Sphinx caches parsed documents inside the output directory unless told
# otherwise, which would make the build cache part of the published site.
"$PYTHON_BIN" -m sphinx -W --keep-going -d docs/build/doctrees -b html docs docs/build/html "$@"
echo "Built docs/build/html/index.html"
