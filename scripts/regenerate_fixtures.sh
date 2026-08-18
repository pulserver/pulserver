#!/bin/bash
# Regenerate every checked-in test fixture:
#
#   tests/python/fixtures/         the zoo corpus (.seq + .bin twins)
#   tests/utils/expected/          the synthetic C-test corpus
#   tests/utils/expected/binary/   the paired text/binary encodings
#
# Deterministic: a second run leaves the tree unchanged, so `git status` after
# running is how you tell whether a change altered any fixture.
#
# Usage:
#   bash scripts/regenerate_fixtures.sh           # every fixture
#   bash scripts/regenerate_fixtures.sh --only=zoo     # the sequence corpus
#   bash scripts/regenerate_fixtures.sh --only=binary  # the binary twins
#
# Environment: PYTHON_BIN selects the interpreter.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ONLY="all"
for arg in "$@"; do
    case "$arg" in
        --only=*) ONLY="${arg#--only=}" ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

case "$ONLY" in
    all|zoo|binary) ;;
    *) echo "--only must be one of: all, zoo, binary" >&2; exit 2 ;;
esac

echo "python: $PYTHON_BIN"

if [ "$ONLY" = "all" ] || [ "$ONLY" = "zoo" ]; then
    section "generate_fixtures.py"
    "$PYTHON_BIN" tests/utils/generate_fixtures.py
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "binary" ]; then
    section "make_binary_fixtures.py"
    "$PYTHON_BIN" tests/utils/make_binary_fixtures.py
fi

echo
echo "Done. Review with 'git status' -- an unchanged tree means nothing moved."
