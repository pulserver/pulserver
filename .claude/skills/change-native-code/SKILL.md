---
name: change-native-code
description: >
  Change the C library (src/c/) or the C++ libraries (src/cpp/), including the
  pybind11 wrappers. Use when editing the parser, the PulSeg representation,
  structure detection, the safety engine, the writers, or the recon-side
  reader — and whenever a C89 or build error needs resolving.
---

# Changing the native code

Two libraries with different rules. `src/c/` runs on the scanner; `src/cpp/` runs
on the host and in the reconstruction.

## src/c/ is ANSI C (C89)

Scanner embedded targets are 32-bit with old toolchains, so the library
compiles as C89 with no dependencies. That means:

- declarations at the top of a block, before any statement — group them in a
  nested `{ ... }` scope with the code that uses them rather than in one
  large block at the top of a long function;
- `/* ... */` comments only;
- no `long long`; `long` may be 32-bit, so use `double` when a
  count can exceed two billion, and integer **microseconds** for time;
- no variable-length arrays, no designated initialisers.

The gate is real and must stay green:

```bash
bash scripts/check_c89_compliance.sh     # -std=c89 -pedantic -Werror -Wall -Wextra
```

It fails on unused variables and unused static functions too, which is how
dead code left behind by a refactor gets caught. Remove it rather than
silencing it.

## src/cpp/ is C++17

`src/cpp/pulseq/` is the design and write path and is on the hot loop for
million-block sequences — measure before adding an allocation per block.
`src/cpp/recon/` is the reconstruction reader and MRD client, where the standard
library is free to use.

## Build and test

```bash
bash scripts/run_tests.sh --only=native   # builds both suites and runs them
bash scripts/run_tests.sh --only=c        # minunit only
bash scripts/run_tests.sh --only=cpp      # GoogleTest only
```

The lanes rebuild from scratch (each build script wipes its build directory),
so do not run two of them concurrently against the same tree.

To rebuild by hand: `bash scripts/build_ctests.sh`,
`bash scripts/build_cpp_tests.sh`.

Changing anything the Python package calls means rebuilding the extension.
Both libraries link into one module, `pulserver._ext`, built from
`src/python/bindings/`:

```bash
pip install -e . --no-build-isolation
```

## What the tests expect of you

- **A differential twin.** Any fast path is asserted equal to the plain
  calculation it replaces, in the same suite — the memoized PNS against the
  exact convolution (`run_pns_memo_equivalence` in
  `tests/ctests/test_safety_grad.c`), the binary reader against the text
  reader, the C SAFE model against upstream's Python. Add one with any new
  optimisation.
- **Byte-for-byte writers.** The text and binary writers are diffed against
  the Python writer's output for the same sequence. A format string is not
  stylistic; changing one changes the file.
- **Structure conformance.** The segmentation the C library derives is
  validated against an independent implementation of the PulSeg rules over
  the whole zoo (`tests/python/test_pulseg_oracle.py`).

## Formatting

`clang-format` with the repository's `.clang-format`; run
`bash scripts/format_and_lint.sh --only=native` to apply it, or with
`--check` to see what would change. The style never aligns continuation lines
against an opening bracket, and once a call does not fit it puts every
argument on its own line.

## Finish by

1. `bash scripts/format_and_lint.sh` — including the C89 compile.
2. `bash scripts/run_tests.sh` — native *and* Python, since the bindings sit
   on top of this code.
3. Reporting the actual test output.
