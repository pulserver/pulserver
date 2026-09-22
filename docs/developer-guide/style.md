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
