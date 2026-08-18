#!/bin/bash
# Format and lint every language in the repository.
#
#   Python   ruff format, then ruff check
#   C / C++  clang-format, then a strict-C89 compile of src/c/
#
# Usage:
#   bash scripts/format_and_lint.sh              # rewrite files in place
#   bash scripts/format_and_lint.sh --check      # report only, exit non-zero
#   bash scripts/format_and_lint.sh --only=python
#   bash scripts/format_and_lint.sh --only=native --check
#   bash scripts/format_and_lint.sh --skip-c89   # leave the C89 compile out
#
# ``--check`` is what CI runs: nothing is written, and any file that would
# change is an error. The default rewrites, which is what pre-commit wants.
#
# The C89 compile needs cmake and a C compiler; without them it is reported as
# skipped rather than failing the run.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CHECK=0
ONLY="all"
SKIP_C89=0

for arg in "$@"; do
    case "$arg" in
        --check) CHECK=1 ;;
        --only=*) ONLY="${arg#--only=}" ;;
        --skip-c89) SKIP_C89=1 ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

case "$ONLY" in
    all|python|native|c|cpp) ;;
    *) echo "--only must be one of: all, python, native, c, cpp" >&2; exit 2 ;;
esac

run_python=0
run_native=0
case "$ONLY" in
    all) run_python=1; run_native=1 ;;
    python) run_python=1 ;;
    native|c|cpp) run_native=1 ;;
esac

STATUS=0
fail() {
    echo "FAIL: $*" >&2
    STATUS=1
}

# ---------------------------------------------------------------- Python ----
# Ruff is configured with `fix = true` in pyproject.toml, so a bare
# `ruff check` rewrites files. --no-fix is what makes the check mode a
# read-only report.
if [ "$run_python" = 1 ]; then
    # The ruff beside the chosen interpreter first, so the version pinned in
    # that environment is the one that decides; then PATH; then the module.
    RUFF=""
    if [ -x "$(dirname "$PYTHON_BIN")/ruff" ]; then
        RUFF="$(dirname "$PYTHON_BIN")/ruff"
    elif command -v ruff >/dev/null; then
        RUFF="$(command -v ruff)"
    elif "$PYTHON_BIN" -m ruff --version >/dev/null 2>&1; then
        RUFF="$PYTHON_BIN -m ruff"
    fi

    if [ -z "$RUFF" ]; then
        fail "ruff not found -- pip install ruff"
    elif [ "$CHECK" = 1 ]; then
        section "ruff format --check"
        $RUFF format --check . || fail "ruff format: files need formatting"
        section "ruff check --no-fix"
        $RUFF check --no-fix . || fail "ruff check: lint errors"
    else
        section "ruff format"
        $RUFF format . || fail "ruff format"
        section "ruff check --fix"
        $RUFF check --fix . || fail "ruff check: errors remain after fixing"
    fi
fi

# ----------------------------------------------------------------- C/C++ ----
native_sources() {
    find src/c src/cpp src/python/bindings tests/ctests tests/cpptests \
        -type f \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \) \
        -not -path '*/build/*' \
        -not -path '*/build314/*' \
        2>/dev/null | sort
}

if [ "$run_native" = 1 ]; then
    if ! command -v clang-format >/dev/null; then
        fail "clang-format not found -- apt install clang-format"
    else
        mapfile -t SOURCES < <(native_sources)
        if [ "${#SOURCES[@]}" -eq 0 ]; then
            fail "no C/C++ sources found"
        elif [ "$CHECK" = 1 ]; then
            section "clang-format --dry-run (${#SOURCES[@]} files)"
            unformatted=()
            for src in "${SOURCES[@]}"; do
                if ! clang-format --dry-run -Werror "$src" >/dev/null 2>&1; then
                    unformatted+=("$src")
                fi
            done
            if [ "${#unformatted[@]}" -gt 0 ]; then
                printf '%s\n' "${unformatted[@]}"
                fail "clang-format: ${#unformatted[@]} file(s) need formatting"
            else
                echo "all ${#SOURCES[@]} files formatted"
            fi
        else
            section "clang-format -i (${#SOURCES[@]} files)"
            clang-format -i "${SOURCES[@]}"
            echo "formatted ${#SOURCES[@]} files"
        fi
    fi

    # The scanner target compiles src/c/ as ANSI C, so a C99-ism that GCC
    # accepts here is a build failure there. This is a lint, not a format,
    # and runs in both modes.
    if [ "$SKIP_C89" = 1 ]; then
        section "C89 compliance (skipped: --skip-c89)"
    elif ! command -v cmake >/dev/null; then
        section "C89 compliance (skipped: cmake not available)"
    else
        section "C89 compliance"
        bash scripts/check_c89_compliance.sh || fail "src/c/ does not compile as C89"
    fi
fi

echo
if [ "$STATUS" = 0 ]; then
    echo "OK"
else
    echo "One or more checks failed."
    if [ "$CHECK" = 1 ]; then
        echo "Run 'bash scripts/format_and_lint.sh' to fix what is fixable."
    fi
fi
exit "$STATUS"
