# User guide

Installing pulserver, running its services and writing the plugins they load.
The concepts behind them are in {doc}`../explanations/index`, and the exact
interfaces, units and defaults in {doc}`../api/index`.

| Page | Task |
| --- | --- |
| {doc}`running` | Run the design calls, the warm design server, the reconstruction proxy and the reconstruction server, and check a sequence plays as written. |
| {doc}`scanner-sequences` | Bind a sequence function to the scanner protocol. |
| {doc}`reconstruction-plugins` | Write a reconstruction plugin and run it offline. |
| {doc}`reconstruction-client` | Send a series to the proxy from a scanner's reconstruction client. |
| {doc}`virtual-scanner` | Scan a phantom through a stored design, the proxy and a reconstruction plugin, without a scanner. |

```{toctree}
:hidden:

running
scanner-sequences
reconstruction-plugins
reconstruction-client
virtual-scanner
```

## Supported platforms

pulserver supports Python 3.10 through 3.14; CI tests both bounds on Linux and
macOS. Windows is not supported: the warm design server forks one process per
call and listens on a Unix socket.

| Platform | Wheel architectures |
| --- | --- |
| Linux | x86-64, glibc (`manylinux`) |
| macOS | Apple silicon |

Each wheel carries the compiled extension `pulserver._ext`. A source build
needs a C compiler, a C++17 compiler, CMake and the Python development headers.

## Installation

```bash
pip install pulserver
```

This installs pypulseqpp, which designs the sequences; bartorch and PyTorch,
which the shipped reconstructions use; blochsim, with which the virtual
scanner simulates the signal and the shipped `fse3d` plugin optimizes its
refocusing trains; the ISMRMRD and DICOM libraries; and brainweb-dl and
websockets, for the virtual scanner's BrainWeb phantom and `pulserver console`.
A reconstruction plugin may import any other engine; the reconstruction proxy
imports bartorch into its spare workers ahead of a series.

The shipped EPI reconstruction also corrects susceptibility distortion when
PyHySCO is installed. PyHySCO is GPL-3.0-only, so it is never installed for
you:

```bash
pip install 'bartorch[pyhysco]'
```

A development installation is described in {doc}`../developer-guide/index`.

## Issue tracker

Report a reproducible defect, a documentation error or a narrowly scoped
feature request on the
[GitHub issue tracker](https://github.com/pulserver/pulserver/issues). Include
the pulserver, pypulseqpp and Python versions, the operating system, the
affected subpackage and the smallest reproducer: a plugin file, a short `.seq`
file, or the design call and its reply.

A defect in sequence design belongs on the
[pypulseqpp tracker](https://github.com/pulserver/pypulseqpp/issues), and one
in a reconstruction algorithm on the
[bartorch tracker](https://github.com/mcencini/bartorch/issues).

## Discussions

[GitHub Discussions](https://github.com/pulserver/pulserver/discussions) is the
public forum for usage questions and broad design ideas.

## Security

The policy is [`SECURITY.md`](https://github.com/pulserver/pulserver/blob/main/SECURITY.md).
Report a vulnerability through
[GitHub private vulnerability reporting](https://github.com/pulserver/pulserver/security/advisories/new),
not through a public issue.

## Safety

Before it writes an IR cache, pulserver runs pypulseqpp's timing, gradient,
PNS, mechanical-resonance and sound pressure checks under the limits the
interpreter sends, and refuses a sequence raising a safety check violation.
SAR, RF coil heating and gradient heating are computed by the vendor's routines
on the scanner, from inputs pulserver provides
({doc}`../explanations/safety-checks`). The checks compute estimates; passing
them does not establish scanner or patient safety.
