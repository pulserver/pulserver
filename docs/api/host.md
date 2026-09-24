# Host daemon

Design sessions for the PSD host processes of one scanner, the revisions they
generate, and the daemon and client that exchange them.

```{eval-rst}
.. currentmodule:: pulserver.host
```

The daemon listens on a Unix socket and runs plugin code in a pool of spawned
worker processes; it is started with

```bash
python -m pulserver.host --base DIR --socket PATH --plugins DIR [--workers N]
```

Sessions and revisions are stored under `<base>/bucket/`, in the layout
described in {doc}`../explanations/sessions`. The `pulserver design` command
answers the same calls without a session, in its own process or in a warm
server's, and stores each design in a {class}`~pulserver.host.DesignStore`
({doc}`../user-guide/running`).

## Daemon and client

| Object | Description |
| --- | --- |
| {obj}`~pulserver.host.HostDaemon` | Unix-socket server answering the commands of every PSD host process on the host. |
| {obj}`~pulserver.host.HostClient` | One session's connection to the daemon, sending commands as a PSD host process does. |
| {obj}`~pulserver.host.HostError` | Raised when the daemon answers a command with `ERROR`. |

## Sessions

| Object | Description |
| --- | --- |
| {obj}`~pulserver.host.SessionKey` | Process ID and start day identifying a PSD host process. |
| {obj}`~pulserver.host.SessionStore` | Every session under `<base>/bucket/`. |
| {obj}`~pulserver.host.Session` | State of one session: plugin, limits, revisions and the current revision. |
| {obj}`~pulserver.host.revision_hash` | Identity of a design: plugin, scanner limits, resolved protocol and the source that designs it. |

## Designs

| Object | Description |
| --- | --- |
| {obj}`~pulserver.host.DesignStore` | The designs under one directory, each in `<id>/` and immutable once written. |
| {obj}`~pulserver.host.design_identity` | SHA-256 of what a design depends on: plugin, limits, resolved protocol and source. |
| {obj}`~pulserver.host.design_id` | Identifier of a design: the first 18 hexadecimal digits of its identity. |
