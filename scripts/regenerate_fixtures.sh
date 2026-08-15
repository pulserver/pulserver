#!/bin/bash
# Regenerate the checked-in .seq/.bin fixture corpus in tests/python/fixtures/.
# Deterministic: a second run leaves the tree unchanged.

set -e

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/mcencini/pulserver-project/.venv/bin/python}"

"$PYTHON_BIN" tests/utils/generate_fixtures.py
