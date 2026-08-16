#!/bin/bash
# Regenerate every checked-in fixture:
#   * tests/python/fixtures/  -- the zoo corpus (.seq + .bin twins)
#   * tests/utils/expected/   -- the synthetic C-test corpus
#   * tests/utils/expected/binary/ -- the paired text/binary encodings
# Deterministic: a second run leaves the tree unchanged.

set -e

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/mcencini/pulserver-project/.venv/bin/python}"

"$PYTHON_BIN" tests/utils/generate_fixtures.py
"$PYTHON_BIN" tests/utils/make_binary_fixtures.py
