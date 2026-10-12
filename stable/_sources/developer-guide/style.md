# Coding style

## Formatting and tests

Ruff formats and lints the Python sources; its configuration and pinned
version are in `pyproject.toml`. Before you report a change complete, run:

```bash
bash scripts/format_and_lint.sh
pytest -q
```

Tests are pytest functions and fixtures, never `unittest.TestCase`, and a test
name states the invariant it protects. Numerical code that can run on CPU and
CUDA is parametrised over both with the `device` fixture, whose CUDA leg skips
without a device.

## Code conventions

`src/c/` is linked into the scanner interpreter and is ANSI C (C89). It calls
nothing in `src/cpp/`, whose passes run only on the host. Docstrings are NumPy
style, and comments and docstrings describe the code as it is, not its
history. `AGENTS.md` states these rules in full.

## Simulation with KomaMRI

`tests/test_koma.py` simulates every fixture and every sequence pypulseqpp
ships with KomaMRI, as designed and as {func}`~pulserver.virtual.export`
writes what its cache plays, and compares the two signals. It runs where
`PULSERVER_KOMA_PROJECT` names a Julia project holding KomaMRI and `julia` is
on the path, and skips elsewhere; `PULSERVER_KOMA_CASES` restricts it to a
comma-separated list of cases. The KomaMRI workflow runs it every night
against pypulseqpp's latest release, and on demand. To run it locally with
Julia 1.10 or later:

```bash
julia --project=tests/koma -e 'using Pkg; Pkg.instantiate()'
PULSERVER_KOMA_PROJECT=tests/koma pytest -q tests/test_koma.py
```

`tests/koma/Project.toml` admits the patch releases of the KomaMRI minor
versions it names; a newer minor version is adopted by editing its `[compat]`
entries.
