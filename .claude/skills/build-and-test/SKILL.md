---
name: build-and-test
description: Build pulserver from this checkout and run its formatter, linter, test suite and documentation build. Use before reporting any change complete, and whenever a test result or a skip has to be quoted rather than assumed.
---

# Build and test

The extension `pulserver._ext` is compiled from `src/cpp/` and `src/c/`, so a
checkout has to be installed before anything imports `pulserver`. Run the whole
sequence and report the results it actually produced.

```bash
pip install -e '.[dev,doc]'
bash scripts/format_and_lint.sh --check
pytest -q -ra
bash scripts/build_docs.sh
```

`scripts/format_and_lint.sh` runs `ruff format` and `ruff check`; without
`--check` it rewrites the files instead of reporting them. It is the entry point
pre-commit and CI use, so a style failure in CI is reproducible here.

## Interpreting the run

- The install needs a C compiler, a C++17 compiler and CMake. `src/c/` is
  compiled with `-std=c90 -pedantic-errors`: a C99 construct is a build error,
  not a warning to silence.
- `tests/test_ir.py` compiles `src/c/` as a 32-bit scanner build and skips
  without a 32-bit toolchain (`gcc-multilib`). Report that skip; it is not a
  pass.
- The CUDA leg of the `device` fixture skips without a device.
- The scanner-sequence plugins under `tests/plugins/` bind arguments of the
  installed pypulseqpp's sequence applications. A failure naming an argument
  `init_sequence` does not take means the plugin and the installed pypulseqpp
  disagree; check `pip show pypulseqpp` against the floor in `pyproject.toml`
  before changing either.
- `tests/test_docs.py` runs the user-guide doctests and holds every public name
  to its API page; a new public name needs a row there.

## Documentation

Documentation changes are validated by the documentation build and
`tests/test_docs.py`, not by reading the source. See the `write-documentation`
skill.
