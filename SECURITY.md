# Security policy

## Reporting a vulnerability

Report privately through
[GitHub's private advisory form](https://github.com/pulserver/pulserver/security/advisories/new).
Please do not open a public issue for a vulnerability.

Include what you would need yourself to reproduce it: the version or commit,
the platform, and the smallest input that triggers it. If the finding involves
a sequence file, attach it — a `.seq` is text and reviewable.

You can expect an acknowledgement within a week, an assessment of severity and
scope after that, and a fix released with the advisory once one is ready.
Credit goes to the reporter unless you ask otherwise.

## Supported versions

Pulserver is pre-1.0. Fixes land on the default branch and go out in the next
release; there are no maintained backport branches yet.

| Version | Supported |
|---|---|
| latest release | yes |
| older releases | no |

## Scope

The parsers are the part most worth attacking, because they read files that
arrive from elsewhere:

- **The C `.seq` parser and the PulSeg conversion** (`src/c/`). C89, no
  dependencies, and it runs on the scanner host. A malformed or hostile `.seq`
  that causes a read outside a buffer, an unchecked allocation, or a crash is
  in scope.
- **The C++ sequence and reconstruction readers** (`src/cpp/`), including the
  binary Pulseq format and the MRD client.
- **The reconstruction server** (`pulserver.recon`), which accepts network
  connections and reads MRD streams.
- **The Nim bridge hosts**, which load a user plugin and answer a console.

## Not vulnerabilities

- **A sequence that is unsafe to play.** Gradient, PNS and acoustic limits are
  checked, but Pulserver's verdict is an estimate that runs before the
  scanner's own. The scanner's predownload check and its hardware monitor are
  authoritative, and a disagreement is a bug report, not a security issue.
- **Running an untrusted sequence plugin.** A plugin is a Python program that
  the operator chose to install; it runs with their privileges by design.
- **Optional dependencies.** `pulserver[distortion]` installs PyHySCO, which
  is GPL-3.0-only and is neither imported nor bundled here. Report issues in
  it to its own project.

## Clinical use

Pulserver is research software. It is not a medical device, carries no
regulatory clearance, and must not be used to make clinical decisions. Any use
on a scanner is the operator's responsibility, under their site's own
approvals.
