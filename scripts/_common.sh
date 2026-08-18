#!/bin/bash
# Shared setup for the scripts in this directory. Source it, do not run it:
#
#   source "$(dirname "$0")/_common.sh"
#
# Provides REPO_ROOT, PYTHON_BIN and a section() helper, and leaves the shell
# in REPO_ROOT so every script takes repo-relative paths.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# PYTHON_BIN from the environment wins, so CI can point at the interpreter it
# set up. Otherwise prefer the project's own virtual environment -- it is the
# one with the package and its test dependencies installed -- then an active
# virtual environment, then whatever python is on PATH.
if [ -z "${PYTHON_BIN:-}" ]; then
    for candidate in \
        "$REPO_ROOT/.venv/bin/python" \
        "$REPO_ROOT/../../../.venv/bin/python" \
        "${VIRTUAL_ENV:-}/bin/python"
    do
        if [ -x "$candidate" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
fi
export PYTHON_BIN

section() {
    printf '\n\033[1m== %s\033[0m\n' "$*"
}
