# User guide

Installing pulserver, running its two services, and writing the plugins they
load. How the components fit together is described in
{doc}`../explanations/index`; exact interfaces are in {doc}`../api/index`.

| Page | Contents |
| --- | --- |
| This page | Supported platforms, installation, issues and security |
| {doc}`running` | The host daemon and the reconstruction proxy |
| {doc}`scanner-sequences` | Binding a pypulseqpp sequence application to the scanner protocol |
| {doc}`reconstruction-plugins` | Writing a reconstruction and running it offline |

```{toctree}
:hidden:

running
scanner-sequences
reconstruction-plugins
```

## Supported platforms

pulserver supports Python 3.10 through 3.13 on Linux and macOS. CI tests the
lower and upper bounds on both. Windows is not supported: the host daemon
listens on a Unix socket.

Published wheels cover:

| Platform | Architectures |
| --- | --- |
| Linux | x86-64, glibc (`manylinux`) |
| macOS | Apple silicon |

A source build compiles the extension `pulserver._ext` and requires a C and a
C++17 compiler, CMake and the Python development headers.

## Installation

```bash
pip install pulserver
```

This installs pypulseqpp, which designs the sequences, and the MRD and DICOM
readers the reconstruction side uses. bartorch is not a dependency: a
reconstruction plugin imports whatever reconstruction engine it needs, and the
reconstruction proxy imports bartorch ahead of a series when it is installed.

Developer installation is documented in {doc}`../developer-guide/index`.

## Issues

Use the [issue tracker](https://github.com/pulserver/pulserver/issues) for a
reproducible defect, a documentation error or a narrowly scoped feature
request. A useful report names the affected subpackage and the pulserver,
pypulseqpp and Python versions, and gives the smallest reproducer: a plugin
file, a short `.seq` file, or the command exchange with the host daemon.

A defect in sequence design belongs to
[pypulseqpp](https://github.com/pulserver/pypulseqpp/issues), and one in a
reconstruction algorithm to [bartorch](https://github.com/mcencini/bartorch/issues).

## Security

The policy is [`SECURITY.md`](https://github.com/pulserver/pulserver/blob/main/SECURITY.md).
Report a vulnerability through
[GitHub private vulnerability reporting](https://github.com/pulserver/pulserver/security/advisories/new),
not through a public issue.
