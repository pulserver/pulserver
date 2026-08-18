#!/bin/bash
# Run the test suite. Every language is driven from one pytest session:
#
#   Python  tests/python/**                       (the bulk of the suite)
#   C       tests/python/native/test_ctests.py    -> the minunit suites
#   C++     tests/python/native/test_cpptests.py  -> the GoogleTest cases
#   Nim     tests/python/native/test_nimtests.py  -> the bridge suite
#
# The native lanes build their binaries on demand (PULSERVER_BUILD_NATIVE=1,
# set here by default) and skip with an explanation when a toolchain is
# missing, so the script runs anywhere and covers whatever the machine can.
#
# Usage:
#   bash scripts/run_tests.sh                  # everything available
#   bash scripts/run_tests.sh --only=python    # skip the C/C++/Nim lanes
#   bash scripts/run_tests.sh --only=native    # only C, C++ and Nim
#   bash scripts/run_tests.sh --only=nim       # only the Nim bridge suite
#   bash scripts/run_tests.sh --coverage       # with coverage.xml, for CI
#   bash scripts/run_tests.sh -k oracle        # any pytest argument passes through
#   bash scripts/run_tests.sh tests/python/safety
#
# Environment: PULSERVER_SKIP_NATIVE=1 and PULSERVER_SKIP_NIM=1 leave a lane
# out; PYTHON_BIN selects the interpreter.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ONLY="all"
COVERAGE=0
PYTEST_ARGS=()
TARGET="tests/python"

for arg in "$@"; do
    case "$arg" in
        --only=*) ONLY="${arg#--only=}" ;;
        --coverage) COVERAGE=1 ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        # An explicit path replaces the default target rather than adding to it.
        tests/*) TARGET=""; PYTEST_ARGS+=("$arg") ;;
        *) PYTEST_ARGS+=("$arg") ;;
    esac
done

NATIVE_LANES="tests/python/native"
case "$ONLY" in
    all) ;;
    python)
        export PULSERVER_SKIP_NATIVE=1 PULSERVER_SKIP_NIM=1
        PYTEST_ARGS+=(--ignore="$NATIVE_LANES")
        ;;
    native)
        [ -n "$TARGET" ] && TARGET="$NATIVE_LANES"
        ;;
    c)
        [ -n "$TARGET" ] && TARGET="$NATIVE_LANES/test_ctests.py"
        ;;
    cpp)
        [ -n "$TARGET" ] && TARGET="$NATIVE_LANES/test_cpptests.py"
        ;;
    nim)
        [ -n "$TARGET" ] && TARGET="$NATIVE_LANES/test_nimtests.py"
        ;;
    *)
        echo "--only must be one of: all, python, native, c, cpp, nim" >&2
        exit 2
        ;;
esac

if [ "$COVERAGE" = 1 ]; then
    PYTEST_ARGS+=(--cov=src/python/pulserver --cov-report=xml)
fi

# The Nim bridge suite installs the bundle from this checkout rather than from
# a release, so a branch is tested against its own sources.
export PULSERVER_TEST_PIP_SPEC="${PULSERVER_TEST_PIP_SPEC:-$REPO_ROOT}"

echo "python:  $PYTHON_BIN"
echo "lanes:   $ONLY"
echo "target:  ${TARGET:-<from arguments>}"

PULSERVER_BUILD_NATIVE="${PULSERVER_BUILD_NATIVE:-1}" \
    exec "$PYTHON_BIN" -m pytest ${TARGET:+"$TARGET"} ${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}
