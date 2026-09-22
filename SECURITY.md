# Security policy

## Reporting a vulnerability

Report privately through
[GitHub's private advisory form](https://github.com/pulserver/pulserver/security/advisories/new).
Please do not open a public issue for a vulnerability.

Include what you would need yourself to reproduce it: the version or commit,
the platform, and the smallest input that triggers it.

You can expect an acknowledgement within a week, an assessment of severity and
scope after that, and a fix released with the advisory once one is ready.
Credit goes to the reporter unless you ask otherwise.

## Supported versions

pulserver is pre-1.0. Fixes land on the default branch and go out in the next
release; there are no maintained backport branches.

| Version | Supported |
|---|---|
| latest release | yes |
| older releases | no |

## Scope

In scope: anything that reads data from outside the process -- the host
daemon's command socket, the reconstruction proxy's TCP port and the MRD
streams it relays, sequence files and IR caches, and any path that accepts a
file or plugin name from a caller.

Out of scope: code in a plugin directory, which both services import and run
by design, so write access to it is equivalent to running code as the service;
resource exhaustion from inputs a caller chose themselves; and behaviour under a
deliberately hostile Python environment.
