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

