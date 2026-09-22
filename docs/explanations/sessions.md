# Design sessions

The host daemon keeps one design session per PSD host process, and stores
everything it generates under `<base>/bucket/`. The reconstruction proxy reads
the same directory to find the design a series was played from.

## Session keys

A session is identified by the process ID of the PSD host process and the day
it started, in days since 1970-01-01, written `<pid>-<day>`
({class}`~pulserver.host.SessionKey`). The interpreter stores both numbers in
float32 slots of the raw-data header, so each must be below 2{sup}`24`, the
largest range of integers a float32 represents exactly.

A session is opened with a scanner-sequence plugin and the scanner limits, and
keeps both for its lifetime: reopening the same key with another plugin or
other limits is refused. A session opened without a plugin only imports
existing sequence files.

## Revisions

Every design a session generates is a revision: a directory
`rev/<n>/` holding the Pulseq files, the IR cache, the resolved protocol and a
`meta.json` record. A revision is written into a staging directory and renamed
into place when complete, and is never modified afterwards. `current` is a
symbolic link to the revision the interpreter plays.

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
        sequence_main.seq the rest of the chain, when there are prescans
        sequence.pseg     IR cache
        resolved.protocol
        meta.json         plugin, reconstruction plugin, limits, hash, files
    queue/                series waiting for a reconstruction slot
```

A revision is identified by the SHA-256 hash of its plugin, limits and resolved
protocol ({func}`~pulserver.host.revision_hash`). A generate request that
resolves to a protocol already generated returns the existing revision and
makes it current rather than designing the sequence again; an import is
identified the same way by the names and contents of the imported files.
Because the hash is taken over the *resolved* protocol, two requests that
differ only in a value the design rounds or replaces, such as a preset, share
a revision.

## What the reconstruction side reads

The raw-data header of a series carries the session key and revision number in
the `pulserver_session` and `pulserver_revision` user parameters. Since a
revision never changes, the proxy reads it once
({class}`~pulserver.vre.RevisionStore`) and keeps it for every later series
played from it.
