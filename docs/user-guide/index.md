# User guide

Installing the package, the platforms it is supported on, running its services
and writing the plugins they load. The components and the representations they
exchange are described in {doc}`../explanations/index`.

## Documentation sections

| Section | Purpose |
| --- | --- |
| This page | Installation, supported platforms, issues and security. |
| {doc}`running` | Starting the host daemon and the reconstruction proxy, and their options. |
| {doc}`scanner-sequences` | Binding a pypulseqpp sequence application to the scanner protocol. |
| {doc}`reconstruction-plugins` | Writing a reconstruction and running it outside the proxy. |
| {doc}`../explanations/index` | Architecture, protocol resolution, design sessions, the scanner IR and raw-data enrichment. |
| {doc}`../examples/index` | Executable examples: protocol resolution, segmentation for the scanner, and enrichment and reconstruction of a simulated series. |
| {doc}`../api/index` | Exact interfaces, units and defaults. |
| {doc}`../developer-guide/index` | Development setup and contribution workflow. |
| {doc}`../misc/index` | Licensing, related projects and contributors. |
| [Source](https://github.com/pulserver/pulserver) | Repository, issues and discussions. |

```{toctree}
:hidden:

running
scanner-sequences
reconstruction-plugins
```

## Prerequisites and supported platforms

pulserver supports Python 3.10 through 3.13. CI tests the lower and upper
bounds on Linux and macOS. Windows is not supported, because the host daemon
listens on a Unix socket.

Published wheels cover:

| Platform | Architectures |
| --- | --- |
| Linux | x86-64, glibc (`manylinux`) |
| macOS | Apple silicon |

The extension `pulserver._ext` is compiled into each wheel. A source build
requires a C compiler, a C++17 compiler, CMake and the Python development
headers.

## Installation

Install the package from PyPI:

```bash
pip install pulserver
```

This installs pypulseqpp, which designs the sequences, and the ISMRMRD and DICOM
libraries the reconstruction side uses. A reconstruction engine is not a
dependency: a reconstruction plugin imports the engine it uses, and the
reconstruction proxy imports bartorch into its spare workers when it is
installed.

Developer installation is documented in {doc}`../developer-guide/index`.

## Reporting issues

Use the [GitHub issue tracker](https://github.com/pulserver/pulserver/issues)
for a reproducible defect, a documentation error, or a narrowly scoped feature
request. A useful report includes the pulserver, pypulseqpp and Python versions,
the operating system, the affected subpackage and the smallest reproducer: a
plugin file, a short `.seq` file, or the command exchange with the host daemon.

A defect in sequence design belongs to the
[pypulseqpp tracker](https://github.com/pulserver/pypulseqpp/issues), and one
in a reconstruction algorithm to the
[bartorch tracker](https://github.com/mcencini/bartorch/issues).

## Discussions

[GitHub Discussions](https://github.com/pulserver/pulserver/discussions) is the
public forum for usage questions and broad design ideas. Reproducible defects
belong in the issue tracker.

## Security

The canonical policy is [`SECURITY.md`](https://github.com/pulserver/pulserver/blob/main/SECURITY.md).
Vulnerabilities must be reported through
[GitHub private vulnerability reporting](https://github.com/pulserver/pulserver/security/advisories/new),
not through a public issue.

## Safety

Pulserver designs, converts and reconstructs; it performs no safety check of
its own. The timing, gradient, PNS, mechanical-resonance and SAR checks of
pypulseqpp compute estimates. Passing these checks does not establish scanner
or patient safety.
