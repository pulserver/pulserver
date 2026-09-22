# Developer guide

Setting up a development checkout, the checks a change has to pass, the
conventions the code follows, and how a release is made.

| Page | Contents |
| --- | --- |
| This page | Setup, checks, conventions, pull requests and releases |
| {doc}`documentation` | Documentation types, where each lives, and how the pages are built |

```{toctree}
:hidden:

documentation
```

## Setup

A development install compiles `pulserver._ext` and needs a C compiler, a
C++17 compiler and CMake.

```bash
git clone https://github.com/pulserver/pulserver.git
cd pulserver
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,doc]'
pre-commit install
```

The test that compiles `src/c/` as a 32-bit scanner build skips without a
32-bit C toolchain; on Debian and Ubuntu, `gcc-multilib` provides one.

## Checks

```bash
bash scripts/format_and_lint.sh --check   # without --check, rewrites in place
pytest -q
bash scripts/build_docs.sh                # warnings are errors
```

CI runs the same formatting script, the tests on Linux and macOS at the lowest
and highest supported Python, and the documentation build. The pre-commit hook
runs the formatting script; if it rewrites a file, stage the file and commit
again.

## Conventions

- `src/c/` is ANSI C (C89) and is compiled with `-std=c90 -pedantic-errors`,
  as a scanner build compiles it. It calls nothing in `src/cpp/`.
- Tests are pytest functions and fixtures, never `unittest.TestCase`. A test
  name states the invariant it protects, so a failure reads as a sentence.
- Anything numerical that can run on CPU and CUDA is parametrised over both
  with the `device` fixture; the CUDA leg skips without a device.
- Docstrings are NumPy style and state what the name and signature do not:
  units, frames, composition order, invariants, side effects and return
  conventions.
- Comments and documentation describe the code as it is. They do not narrate
  its history or name fixed bugs.

`AGENTS.md` states these rules in full for human and automated contributors.

## Pull requests

Open pull requests against `main`. The template asks what changed, why, the
exact validation commands and their results, and the documentation impact.
Participation is governed by the
[code of conduct](https://github.com/pulserver/pulserver/blob/main/CODE_OF_CONDUCT.md).

## Releases

Versions come from Git tags through `setuptools_scm`. Pushing a tag
`vX.Y.Z` builds the wheels and the source distribution, publishes them to PyPI
through trusted publishing, creates a GitHub release with Sigstore signatures,
and publishes the documentation for that version.
