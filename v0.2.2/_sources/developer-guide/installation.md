# Development setup

## Prerequisites

You need Git, Python 3.10 to 3.14, a C compiler, a C++17 compiler and CMake,
on Linux or macOS. Windows is not supported: the warm design server forks per
call and listens on a Unix socket.

The test that compiles `src/c/` as a 32-bit scanner build needs a 32-bit C
toolchain, which `gcc-multilib` provides on Debian and Ubuntu; without one it
skips.

## Editable installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,doc]'
```

The `dev` extra holds the test, lint and pre-commit tools, with Ruff pinned to
one version. The `doc` extra holds Sphinx, sphinx-gallery and matplotlib.

The installation compiles the extension `pulserver._ext` from `src/cpp/` and
`src/c/`; `src/c/` is compiled with `-std=c90 -pedantic-errors`, as a scanner
build compiles it. The extension is not rebuilt on import. After you change
`src/cpp/` or `src/c/`, or switch to a branch that changes them, run the
installation again; otherwise the tests run against the previous build.

## Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

The local hook runs `scripts/format_and_lint.sh`, which formats and lints in
place; the Style workflow runs the same script with `--check`. The whitespace
hooks skip the sequence files under `tests/fixtures/`.

`git commit --no-verify` bypasses every hook for one commit, and
`SKIP=<hook-id> git commit` bypasses one. CI runs the same checks, so a bypass
does not let a failing check through.

## Documentation build

```bash
bash scripts/build_docs.sh
```

The script builds the HTML into `docs/build/html` and executes every gallery
script while it does; a Sphinx warning fails the build.
