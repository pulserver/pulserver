# scripts/

Four entry points cover everyday work. Each takes `--help`, resolves the
interpreter the same way (`PYTHON_BIN`, else the project `.venv`, else
`python3`), and runs from anywhere.

| | |
|---|---|
| `bash scripts/build_docs.sh` | Build the Sphinx documentation. `--clean`, `--strict`. |
| `bash scripts/format_and_lint.sh` | Format and lint Python (ruff), C and C++ (clang-format), and compile `src/c/` as C89. `--check` reports without rewriting; `--only=python\|native\|c\|cpp`; `--skip-c89`. |
| `bash scripts/regenerate_fixtures.sh` | Rebuild every checked-in test fixture. `--only=zoo\|binary`. |
| `bash scripts/run_tests.sh` | Run the suite: Python, C, C++ and Nim, from one pytest session. `--only=python\|native\|c\|cpp\|nim`, `--coverage`, plus any pytest argument. |

Two of these are shared with automation, so there is one definition of each:

- `.pre-commit-config.yaml` runs `format_and_lint.sh --skip-c89`, and
  `.github/workflows/style.yml` runs `format_and_lint.sh --check`. A commit
  that passes locally cannot fail CI on style.
- `.github/workflows/test-ci.yml` runs `run_tests.sh` once per lane, so a
  failing job names the language it came from.

## The pieces underneath

`build_ctests.sh`, `run_ctests.sh`, `build_cpp_tests.sh`, `run_cpp_tests.sh`
and their `build_and_run_*` pairs configure and drive the native suites
directly; `run_tests.sh` reaches them through the pytest lanes and builds on
demand, so they are only needed to rebuild by hand. `check_c89_compliance.sh`
is the C89 compile `format_and_lint.sh` calls. `make_installer.sh` builds the
distributable bundle, and `_common.sh` is sourced by the four entry points
rather than run.
