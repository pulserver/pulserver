# Contributing to pulserver

The [Developer Guide](https://pulserver.github.io/pulserver/latest/developer-guide/index.html)
documents the toolchain, editable installation, coding and documentation
conventions, pre-commit hooks, pull-request workflow and releases.

A quick development setup is:

```bash
git clone https://github.com/YOUR-USER/pulserver.git
cd pulserver
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev,doc]'
pre-commit install
```

Before opening a pull request:

```bash
bash scripts/format_and_lint.sh --check
pytest -q
bash scripts/build_docs.sh
```

Open the pull request against [`pulserver/pulserver:main`](https://github.com/pulserver/pulserver/compare).
Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
