# Designs and the design store

A scanner sequence is validated on every protocol edit and generated when the
scan is prepared, often many times for one protocol, and a generated design
has to remain available, unchanged, for as long as data acquired with it may
be reconstructed. Pulserver stores each design by its content: a design is
identified by what it depends on, and a request that resolves to a stored
design returns it instead of designing again. No design call keeps state
between calls; the design store is all that persists.

## Design identity

A generated design depends on the scanner-sequence plugin, the code that
designs with it, the scanner limits and the resolved protocol. Its identity is
the SHA-256 hash of these ({func}`~pulserver.host.design_identity`):

- the plugin name;
- a digest of the plugin file, the source file of the application it binds,
  and the installed versions of pypulseqpp and pulserver;
- the scanner limits, conversion options and check limits, with the contents
  of the VOP file they name;
- the resolved protocol, prescription included.

A changed plugin, application or package gives a new design for an unchanged
protocol, and so does a VOP file rewritten at the same path. Because the hash
is computed over the resolved protocol, two requests that differ only in a
value the design replaces, such as a preset and the time it resolves to,
identify the same design ({doc}`protocol`). A design is made from its resolved
protocol: a request that resolves to itself, such as the value block of a
`validate` reply sent back, constructs its application once, and one that resolves to other values
is constructed again from those, so the files of a design are a function of
its identity.

An imported chain is identified by the names and contents of its files, the
limits and the prescription of the import, offset and rotation.

## Design identifier

The identifier of a design is the first 18 hexadecimal digits of its identity,
72 bits ({func}`~pulserver.host.design_id`). The interpreter records the
numbers it passes to the reconstruction client in float32 fields of the
raw-data header, and a float32 represents every integer below $2^{24}$
exactly, so the identifier takes three of them, of six digits each. The store
refuses a design whose identifier already holds another identity.

## The store

The store is a directory with one subdirectory per design:

```text
designs/
  3f9c0a1b2c3d4e5f60/
    sequence.seq          first file of the NextSequence chain, binary Pulseq
    sequence_main.seq     the remaining files, when there are prescans
    sequence.pseg         IR cache
    resolved.protocol
    manifest.json         identity, plugin, reconstruction plugin, limits,
                          package versions, SHA-256 of every file
```

A design is written into a staging directory in the store and renamed into
place once complete, so a reader sees it whole or not at all, and two processes
that write one design leave one directory. A design is not modified afterwards.
Designs are removed only by pruning ({doc}`../user-guide/running`), least
recently used first; a design is used when it is written or found again.

## Access from the reconstruction side

The raw-data header of a series names its design in the `pulserver_design`
user parameter ({doc}`../user-guide/reconstruction-client`). A design is
immutable, so the reconstruction proxy reads and tabulates it once and reuses
the result for later series acquired with it, keeping the designs it read most
recently ({class}`~pulserver.vre.DesignCache`).

When the design calls and the proxy run on different computers, the proxy
reads either a store both can reach or a store of its own, which its design
intake fills with the designs the calls push ({class}`~pulserver.vre.DesignIntake`).
A pushed design is carried as a bundle of the files of its directory, and the
intake stores it only when the manifest's identity is a SHA-256, its identifier
is that of the identity and the one the push names, and every file has the
SHA-256 the manifest records, so the two stores hold the same files under the
same identifier. A design is pushed before its identifier is replied, and a
design that cannot be pushed fails the call, so the interpreter plays no design
the proxy's store lacks.

## See also

* {doc}`../user-guide/running` — the design calls, the warm server, pushing and pruning.
* {doc}`../api/host` — the design calls and the store.
* {doc}`protocol` — the resolution of a request into the protocol a design plays.
