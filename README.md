# Pulserver

[![Tests](https://github.com/pulserver/pulserver/actions/workflows/test-ci.yml/badge.svg)](https://github.com/pulserver/pulserver/actions/workflows/test-ci.yml)
[![Style](https://github.com/pulserver/pulserver/actions/workflows/style.yml/badge.svg)](https://github.com/pulserver/pulserver/actions/workflows/style.yml)
[![Documentation](https://readthedocs.org/projects/pulserver/badge/?version=latest)](https://pulserver.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)

Pulserver takes a vendor-neutral [Pulseq](https://pulseq.github.io) sequence
from design to acquisition. It builds the sequence in Python, checks it
against the hardware limits a scanner enforces before it will play anything,
hands an interpreter the structure it needs to actually play it, and gives the
reconstruction the description it needs to interpret what came back.

A `.seq` file says *what to play*. A scanner needs rather more before it will
play it, and a reconstruction needs rather more before it can use the result.
Pulserver supplies the missing pieces.

## Install

```bash
pip install pulserver
```

Wheels ship the compiled C and C++ extensions, so no compiler, CMake or SDK is
needed on the target machine. One install covers both sides of the scanner —
sequence design and reconstruction — because a system that does both should
not carry two Python environments.

Two optional extras:

```bash
pip install "pulserver[cuda]"        # GPU reconstruction (CUFINUFFT, Linux)
pip install "pulserver[distortion]"  # PyHySCO, GPL-3.0-only, opt-in
```

## A first sequence

```python
from pulserver.app import gre2D_sequence

seq = gre2D_sequence(n_x=256, n_y=256, n_slices=16)

ok, message = seq.check_hardware_limits()  # amplitude, slew, continuity
seq.declare_tr()  # the repeating unit, from the content
seq.write("gre.seq")
```

`seq` is a drop-in [PyPulseq](https://github.com/imr-framework/pypulseq)
`Sequence` — the same functions and the same signatures, so an existing script
runs against it unchanged — with a C++ core underneath. A protocol-scale
MPRAGE of about two million blocks designs in a few seconds from ordinary
Python.

## What it does that a Pulseq writer does not

| | |
|---|---|
| **Structure, derived** | The repeating unit (the TR) and the reusable segments are *detected from the block content*, not annotated by hand. That is what makes a `.seq` playable on hardware that prepares a segment once and replays it, and what makes whole-scan safety checks affordable. |
| **Safety, before download** | Gradient amplitude, slew and continuity always; PNS (Irnich and SAFE) and mechanical resonance when the site supplies the hardware model. The same compiled engine runs at design time and on the scanner, so the two answers cannot disagree. |
| **Interactive throughput** | Design, conversion and the safety analyses are fast enough to answer a console on every protocol edit. The fast paths are held equal to the plain calculations by tests, not by assertion. |
| **A reconstruction that knows what it got** | Encoding counters, k-space trajectory, echo positions and a sequence description travel with the data as an MRD stream, and an off-isocentre prescription is demodulated on the way. |

## Three interfaces, one representation

- **Python** — design sequences from readout, excitation and preparation
  modules; write `.seq` files; run the safety checks; reconstruct.
- **C** — the scanner-side library: parse a `.seq`, derive its structure, gate
  it against the hardware, replay it block by block. C89, no dependencies.
- **C++** — the same library with RAII types, plus the reconstruction-side
  reader that turns a `.seq` chain into trajectories, labels and a sequence
  description.

## Documentation

[pulserver.readthedocs.io](https://pulserver.readthedocs.io)

- **Getting started** builds a gradient echo, checks it, and looks at what the
  interpreter makes of it.
- **Explanations** cover the sequence model, the safety physics, the
  performance budget, and the sequence zoo that is the evidence for all of it.
- **Examples** are worked code for each of the three interfaces.

## Development

```bash
git clone https://github.com/pulserver/pulserver.git
cd pulserver
pip install -e ".[dev]"
pre-commit install
```

Four scripts cover everyday work; each takes `--help`.

```bash
bash scripts/run_tests.sh            # Python, C, C++ and Nim, from one pytest session
bash scripts/format_and_lint.sh      # ruff, clang-format, and a strict-C89 compile
bash scripts/build_docs.sh           # the Sphinx documentation
bash scripts/regenerate_fixtures.sh  # every checked-in test fixture
```

The pre-commit hook and the CI workflows run those same scripts, so a change
that passes locally passes in CI. See [`scripts/README.md`](scripts/README.md).

Contributions are welcome — open an
[issue](https://github.com/pulserver/pulserver/issues) or a
[discussion](https://github.com/pulserver/pulserver/discussions) first if the
change is substantial. Security reports go through
[SECURITY.md](SECURITY.md).

## Status and scope

Pulserver is research software, and pre-1.0: interfaces may still change. It
is **not a medical device** and carries no regulatory clearance. Its safety
checks run *before* the scanner's own, and never in place of them — the
scanner's predownload gate and its hardware monitor remain authoritative.

## License

MIT — see [LICENSE.txt](LICENSE.txt). The optional `distortion` extra installs
PyHySCO, which is GPL-3.0-only; it is driven as an external executable and is
neither imported nor bundled here.
