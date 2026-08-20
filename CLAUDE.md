# Working in this repository

Guidance for coding agents. `AGENTS.md`, `GEMINI.md` and
`.github/copilot-instructions.md` are symbolic links to this file, so every
assistant reads the same instructions and there is one place to change them.

## What this is

Pulserver takes a [Pulseq](https://pulseq.github.io) `.seq` sequence from
design to acquisition: it builds sequences in Python, checks them against the
hardware limits a scanner enforces, derives the structure an interpreter needs
to play them, and streams the reconstruction what it needs to interpret the
data. Read `docs/explanations/` before changing anything conceptual — the
sequence model, the safety physics and the performance strategy are all
written down there.

## Layout

All source lives under `src/`, one directory per language, and each layer is
a superset of the one below it rather than a reimplementation of it.

| Path | What lives there |
|---|---|
| `src/c/` | The scanner-side C library: `.seq` parsing, the PulSeg representation, structure detection, the safety engine. **C89, no dependencies.** |
| `src/cpp/` | C++17: `pulseqpp` (the design/write path, `src/cpp/pulseq/`) and the reconstruction-side reader and MRD client (`src/cpp/recon/`). It links `src/c` rather than restating it. |
| `src/python/pulserver/` | The Python package. `pypulseq/` is a drop-in PyPulseq replacement over the C++ core; `design/` is the module toolbox; `recon/` is the reconstruction stack. |
| `src/python/bindings/` | The pybind11 sources. They build into one extension module, `pulserver._ext`, whose submodules are `pulseg`, `pulseqpp`, `arbgrad`, `sampling` and `recon_cpu`. |
| `src/nim/` | The Nim hosts that let a console drive a Python or MATLAB plugin. |
| `examples/sequence/`, `examples/recon/` | The sequence zoo and its reconstruction plugins. Installed as `pulserver.app.sequence.*` and `pulserver.app.recon.*`, so they are shipped code, not samples — the deliberate exception to "source lives under `src/`". |
| `tests/` | `ctests/` (minunit), `cpptests/` (GoogleTest), `python/` (pytest, including the native lanes), `nim/`, and `utils/` with the fixture generators. |
| `docs/` | Sphinx sources. `_docs/` is a superseded copy — do not add to it. |
| `scripts/` | The four entry points below, plus the build steps they call. |

## The four commands

Everything routes through these; pre-commit and CI call the same scripts, so
what passes locally passes in CI. Each takes `--help`.

```bash
bash scripts/run_tests.sh            # Python, C, C++ and Nim in one pytest session
bash scripts/format_and_lint.sh      # ruff, clang-format, strict-C89 compile
bash scripts/build_docs.sh           # Sphinx
bash scripts/regenerate_fixtures.sh  # every checked-in fixture
```

Useful selectors: `run_tests.sh --only=python|native|c|cpp|nim`, any pytest
argument passes through (`-k`, a path); `format_and_lint.sh --check` reports
without rewriting.

**Run the tests and report the actual output.** A change is not complete
because it looks right. If the native toolchain is missing, those lanes skip
themselves and say so — that is not a pass.

## Language rules

**`src/c/` is ANSI C (C89).** Declarations at the top of a block, no `//`
comments, no mixed declarations and code, no `long long`. Scanner embedded
targets are 32-bit with old toolchains, so times are integer microseconds and
a count that can exceed two billion goes in a `double`.
`format_and_lint.sh` compiles it with `-std=c89 -pedantic -Werror` and that
gate must stay green.

**`src/cpp/` is C++17.** The recon side may use the standard library freely; the
design side is on the hot path for million-block sequences, so measure before
adding an allocation per block.

**Python targets 3.11+.** The package has one dependency set — there is no
design-only or recon-only install, because a scanner runs both and two
environments meant two copies of Torch. Only three extras exist: `cuda`,
`distortion` (GPL, opt-in) and `dev`.

## Comments and docstrings

Write for someone reading the code as it is now, who has no memory of any
earlier version of it. **Never** write text whose subject is the history of
the code. Banned in comments, docstrings and prose docs alike:

- "used to", "was once", "no longer", "previously", "now that", "this
  replaces", "the old X", "before the fix"
- justifying the present shape by contrast with a shape that is gone
- naming a bug that has been fixed, or the session that fixed it
- restating what the code plainly says

A docstring carries what a caller needs: one line of what, then Parameters,
Returns, Raises. A comment earns its place only by explaining a non-obvious
algorithm or a choice a reader would otherwise undo — and even then, prefer a
well-named function or a test whose name states the invariant, because those
cannot go stale silently. When tempted to explain *why not the other way*,
write a test instead.

Deleting an outdated comment is always correct; rewriting one to describe the
change is not.

## Tests

- **Name the invariant.** `test_the_worst_case_tr_bounds_every_instance_it_stands_for`,
  not `test_tr_2`. The name is the specification.
- **A fast path is asserted equal to the plain one.** Every optimisation here
  has a differential test against the calculation it replaces — the memoized
  PNS against the exact convolution, the C SAFE model against upstream's
  Python, the drawn resonance lines against the predownload verdict. Add one
  with any new fast path; a fast answer that disagrees is a different check,
  not an optimisation.
- **`tr=None` is upstream PyPulseq to the bit.** Any change to the analysis
  surface must keep a PyPulseq script getting PyPulseq's numbers.
- Fixtures are generated, deterministic and checked in. After changing
  anything that alters them, run `regenerate_fixtures.sh` and review the diff —
  an unchanged tree means nothing moved.

## Conventions that bite

- **Do not annotate what can be derived.** Segmentation and the TR are
  detected from block content, never from a `TRID` label. An annotation is a
  second source of truth that can disagree with the sequence.
- **Safety verdicts are estimates that run before the scanner's.** They never
  replace the scanner's predownload gate or its hardware monitor. Do not write
  documentation or messages that imply otherwise.
- **`ruff` is configured with `fix = true`**, so a bare `ruff check` rewrites
  files. Use `--no-fix` to inspect without changing anything.
- **Generated and compiled output is not tracked.** Wheels, object files,
  `__pycache__`, rendered docs and benchmark artifacts are all ignored; see
  `.gitignore`. Test fixtures *are* tracked, deliberately.
- **The examples are shipped code.** A change to `examples/sequence/*` or
  `examples/recon/*` changes the installed `pulserver.app` namespace, and the
  zoo tests hold it.
- **A zoo module is one complete plugin and nothing else.** `examples/recon/*`
  holds one `ReconPlugin` subclass, its `PLUGIN`, and the three hooks — never a
  module-level helper, a private method, or a module demonstrating one step.
  A step a plugin needs is a name in `pulserver.recon`, general and high-level
  enough to compose directly in a hook; a local subroutine hides which of the
  code is the mandatory hook. `examples/sequence/*` holds whole sequences on
  the same terms.
- **The three recon hooks divide the work the same way every time.**
  `startup` lays out the buffers the header's encoding spaces describe.
  `receive` places each acquisition and, reading its flags, routes the
  boundaries it closes to a named branch — the sorting and the routing both
  live there. `recon` takes that branch name and holds the reconstruction of
  each branch over buffers that are already filled; it never sorts and never
  decides when it runs.

## Before you finish

1. `bash scripts/format_and_lint.sh` — clean.
2. `bash scripts/run_tests.sh` — and read the output.
3. Documentation updated if behaviour a user sees changed.
4. Say plainly what you ran, what passed, and what you did not check.
