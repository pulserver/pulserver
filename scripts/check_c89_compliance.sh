#!/bin/bash
# check_c89_compliance.sh -- verify that pulseqlib compiles in strict C89 mode.
#
# Usage:  bash scripts/check_c89_compliance.sh
# Returns 0 (PASS) or 1 (FAIL).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/c89_check"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Checking C89 compliance of pulseqlib..."

# Configure with strict C89 flags injected via CMAKE_C_FLAGS
cmake -S "$REPO_ROOT/csrc" -B "$BUILD_DIR" \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_C_FLAGS="-std=c89 -pedantic -Werror -Wall -Wextra" \
    -DCMAKE_BUILD_TYPE=Release \
    > "$BUILD_DIR/cmake_out.log" 2>&1

if cmake --build "$BUILD_DIR" > "$BUILD_DIR/build_out.log" 2>&1; then
    echo "PASS: pulseqlib compiles cleanly in C89 mode."
    rm -rf "$BUILD_DIR"
    exit 0
else
    echo "FAIL: pulseqlib does not compile in C89 mode."
    echo "Build log:"
    cat "$BUILD_DIR/build_out.log"
    rm -rf "$BUILD_DIR"
    exit 1
fi
