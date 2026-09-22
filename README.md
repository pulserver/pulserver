[![Tests](https://github.com/pulserver/pulserver/actions/workflows/test-ci.yml/badge.svg)](https://github.com/pulserver/pulserver/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/pulserver/pulserver/branch/main/graph/badge.svg)](https://codecov.io/gh/pulserver/pulserver)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs: stable](https://img.shields.io/badge/docs-stable-2b76ad)](https://pulserver.github.io/pulserver/stable/)
[![Docs: latest](https://img.shields.io/badge/docs-latest-6b7684)](https://pulserver.github.io/pulserver/latest/)
[![PyPI](https://img.shields.io/pypi/v/pulserver.svg)](https://pypi.org/project/pulserver/)
[![Python](https://img.shields.io/pypi/pyversions/pulserver.svg)](https://pypi.org/project/pulserver/)
[![License: MIT](https://img.shields.io/badge/license-MIT-ffbd28.svg)](https://github.com/pulserver/pulserver/blob/main/LICENSE)

# pulserver

Orchestrator for MR acquisitions: sequence design, scanner preparation,
reconstruction and real-time feedback.

Pulserver plays Pulseq sequences on an MR scanner through the scanner's
interpreter and returns the reconstructed images to the console. Sequence
design is [pypulseqpp](https://github.com/pulserver/pypulseqpp) and
reconstruction is [bartorch](https://github.com/mcencini/bartorch), or whatever
a reconstruction plugin imports; pulserver connects them to the scanner.

- **Host daemon** (`python -m pulserver.host`). Resolves each protocol the
  operator edits into the protocol the sequence will play, writes the design,
  and converts it into the segmented binary cache the interpreter loads.
- **Reconstruction proxy** (`python -m pulserver.vre`). Receives the MRD stream
  of each series, fills in the counters, flags, encoding spaces and trajectory
  from the sequence that played it, and runs a reconstruction plugin in a
  worker process.
- **Plugins.** A scanner sequence binds a pypulseqpp sequence application to
  the scanner protocol; a reconstruction plugin runs over a live stream, an
  MRD file or an assembled bucket through the same hooks.
- **Cache reader** (`src/c/`). The ANSI C library the scanner interpreter links
  to load the cache.

## Installation

```bash
pip install pulserver
```

Wheels are published for Linux x86-64 and macOS on Apple silicon, for Python
3.10 to 3.13.

## Documentation

The [user guide](https://pulserver.github.io/pulserver/latest/user-guide/index.html)
covers installation, running the two services and writing plugins; the
[explanations](https://pulserver.github.io/pulserver/latest/explanations/index.html)
describe the components, design sessions, the IR cache and the reconstruction
side. Every version of the documentation is published at
<https://pulserver.github.io/pulserver/>.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, except vendored third-party components, which keep their own licences; see
[`LICENSES/`](LICENSES/).
