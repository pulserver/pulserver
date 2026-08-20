#!/bin/bash
# Build the Sphinx documentation.
#
# Usage:
#   bash scripts/build_docs.sh            # HTML into docs/_build/html
#   bash scripts/build_docs.sh --clean    # discard docs/_build first
#   bash scripts/build_docs.sh --strict   # turn Sphinx warnings into errors
#
# The interpreter needs sphinx, myst-parser and sphinx-book-theme, plus this
# package importable (conf.py puts ./python on the path itself).
#
# The C and C++ reference additionally needs doxygen, which is a system
# package rather than a Python one. Without it those pages render their prose
# and say so, and the rest of the build is unaffected.
#
# Environment: PYTHON_BIN selects the interpreter.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DOCS="$REPO_ROOT/docs"
OUT="$DOCS/_build/html"
SPHINX_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --clean) rm -rf "$DOCS/_build" ;;
        --strict) SPHINX_ARGS+=(-W --keep-going) ;;
        -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

echo "python: $PYTHON_BIN"

section "sphinx-build"
"$PYTHON_BIN" -m sphinx -b html ${SPHINX_ARGS[@]+"${SPHINX_ARGS[@]}"} "$DOCS" "$OUT"

echo
echo "Done: $OUT/index.html"
