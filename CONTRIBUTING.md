# Contributing to pulserver

The [developer guide](https://pulserver.github.io/pulserver/latest/developer-guide/index.html)
documents the toolchain, the checks, the coding and documentation conventions
and the release procedure.

A development setup needs a C compiler, a C++17 compiler and CMake:

```bash
git clone https://github.com/pulserver/pulserver.git
cd pulserver
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,doc]'
pre-commit install
```

Before opening a pull request:

```bash
bash scripts/format_and_lint.sh --check
pytest -q
bash scripts/build_docs.sh
```

Open the pull request against [`main`](https://github.com/pulserver/pulserver/compare).
Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
