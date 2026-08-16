#!/bin/bash
# Run the whole test suite through pytest: the Python tests plus the native
# lanes that drive every compiled language from the same session --
#
#   * C     -- the minunit suites  (tests/python/native/test_ctests.py)
#   * C++   -- the GoogleTest cases (tests/python/native/test_cpptests.py)
#   * Nim   -- the bridge suite     (tests/python/native/test_nimtests.py,
#              skipped when no Nim toolchain is on PATH)
#
# Native binaries are built on demand (PULSERVER_BUILD_NATIVE=1); pass
# PULSERVER_SKIP_NATIVE=1 / PULSERVER_SKIP_NIM=1 to leave a lane out.
#
# Usage:
#   bash scripts/run_tests.sh                 # everything
#   bash scripts/run_tests.sh -k oracle       # any pytest filter passes through
#   bash scripts/run_tests.sh tests/python/safety

set -e

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/mcencini/pulserver-project/.venv/bin/python}"

TARGET="tests/python"
for arg in "$@"; do
    case "$arg" in
        tests/*) TARGET="" ;;
    esac
done

PULSERVER_BUILD_NATIVE="${PULSERVER_BUILD_NATIVE:-1}" \
    "$PYTHON_BIN" -m pytest $TARGET "$@"
