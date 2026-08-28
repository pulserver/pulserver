#!/bin/bash
# Compile the C examples.
#
# They are rendered into the documentation from source, so a broken one is a
# broken page. Building them is what keeps the two in step.
#
# Usage:
#   bash scripts/build_examples.sh            # build into build/examples
#   bash scripts/build_examples.sh --clean    # discard the build tree first
#
# The examples are held to the same -std=c89 -pedantic the library is: a
# vendor's toolchain is the reason they exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD="$REPOSITORY/build/examples-build"

for argument in "$@"; do
    case "$argument" in
        --clean) rm -rf "$BUILD" ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown argument: $argument" >&2; exit 2 ;;
    esac
done

cmake -S "$REPOSITORY/examples/c" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release > /dev/null

cmake --build "$BUILD" -j"$(nproc 2>/dev/null || echo 4)"

echo
echo "examples built into $BUILD/examples"
