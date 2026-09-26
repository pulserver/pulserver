# Coding style

Ruff supplies Python formatting and linting; configuration and the pinned
version are in `pyproject.toml`. Run:

```bash
bash scripts/format_and_lint.sh
pytest -q
```

Use pytest functions and fixtures rather than `unittest.TestCase`. Test names
state the invariant they protect. Numerical code that can run on CPU and CUDA
is parametrised over both with the `device` fixture, whose CUDA leg skips
without a device.

`src/c/` is linked into the scanner interpreter and is ANSI C (C89). It calls
nothing in `src/cpp/`, whose passes run only on the host. Docstrings are NumPy
style; comments and docstrings describe the code as it is, not its history.
`AGENTS.md` states these rules in full.

## Simulation with KomaMRI

`tests/test_koma.py` simulates every fixture and every sequence pypulseqpp
ships with KomaMRI, as designed and as {func}`~pulserver.virtual.export`
writes what its cache plays, and compares the two signals. It runs where
`PULSERVER_KOMA_PROJECT` names a Julia project holding KomaMRI and `julia` is
on the path, and skips elsewhere. The KomaMRI workflow runs it every night and
on a pull request that changes the IR, the export or the requirements. To run
it locally with Julia 1.10 or later:

```bash
julia --project=tests/koma -e 'using Pkg; Pkg.instantiate()'
PULSERVER_KOMA_PROJECT=tests/koma pytest -q tests/test_koma.py
```

`tests/koma/Project.toml` admits the patch releases of the KomaMRI minor
versions it names; a newer minor version is adopted by editing its `[compat]`
entries.
