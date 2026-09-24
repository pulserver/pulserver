# Design sessions and revisions

A running scanner sequence is edited, validated and prepared many times, and
several can be open at once on one scanner. Each generated design has to remain
available, unchanged, for as long as data acquired with it may still be
reconstructed. The host daemon therefore keeps one design session per PSD host
process and stores every design it generates as an immutable revision, in a
directory, the *bucket*, that the reconstruction proxy also reads.

## Session keys

A session is identified by the process ID of the PSD host process and the day
the process started, in days since 1970-01-01, and written `<pid>-<day>`
({class}`~pulserver.host.SessionKey`). The interpreter records both numbers in
float32 fields of the raw-data header. A float32 represents every integer below
$2^{24}$ exactly, so both are required to be below that bound.

A session is opened with a scanner-sequence plugin and the scanner limits and
retains both: reopening a key with another plugin or other limits is refused.
A session opened without a plugin only imports existing sequence files.

## Revisions

A revision is a directory `rev/<n>/` holding the Pulseq files of one design,
their IR cache, the resolved protocol and a `meta.json` record naming the
plugin and its source digest, the reconstruction plugin, the limits and the
files. A revision is
written into a staging directory and renamed into place once complete, and is
not modified afterwards. `current` is a symbolic link to the revision the
interpreter plays.

```text
bucket/
  <pid>-<day>/
    session.json          plugin, limits, revisions by hash, current revision
    protocol              the latest VALIDATE request
    current -> rev/2
    rev/
      1/
      2/
        sequence.seq      first file of the NextSequence chain
        sequence_main.seq the remaining files, when there are prescans
        sequence.pseg     IR cache
        resolved.protocol
        meta.json
    queue/                series waiting for a reconstruction slot
```

## Revision identity

A generated revision is identified by the SHA-256 hash of its plugin, scanner
limits, resolved protocol and source ({func}`~pulserver.host.revision_hash`).
The source is a digest of the plugin file, the source file of the application
it binds, and the installed versions of pypulseqpp and pulserver. A `GENERATE`
request whose protocol resolves to one already generated from the same source
returns the existing revision and makes it current; the sequence is not
designed again. A changed plugin, application or package designs a new
revision, even for an unchanged protocol.
Because the hash is computed over the resolved protocol, two requests that
differ only in a value the design replaces, such as a preset and the time it
resolves to, identify the same revision ({doc}`protocol`). An imported revision
is identified by the names and contents of the imported files and the
prescription of the import, offset and rotation. The prescription of a
generated revision is part of its resolved protocol. Where the limits name a
VOP file, the file's contents are part of the identity as well as its path, so
a design checked against one set of VOPs is not returned for another written to
the same file.

## Access from the reconstruction side

The raw-data header of a series records the session key and the revision number
in the `pulserver_session` and `pulserver_revision` user parameters. Since a
revision is immutable, the reconstruction proxy reads and tabulates it once and
reuses the result for later series acquired with it, keeping the revisions it
read most recently ({class}`~pulserver.vre.RevisionStore`). When the two
services run on different computers, the bucket is a directory both can
access.

## See also

* {doc}`../user-guide/running` — starting the host daemon and its client.
* {doc}`../api/host` — sessions, revisions and the daemon interface.
