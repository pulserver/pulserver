# Architecture

```{admonition} TL;DR
:class: tldr

- pulserver runs as two services beside the scanner. The design calls on the
  scanner host answer the scanner UI and produce what the scanner plays. The
  reconstruction proxy on the reconstruction computer receives the raw data and
  runs your reconstruction.
- Every design is stored once, under the SHA-256 of everything it depends on,
  and is never modified. A request that resolves to a stored design returns it.
- The scanner's interpreter only presents the protocol, plays the IR cache and
  streams the raw data. Sequence design is pypulseqpp's job, and the
  reconstruction is the engine your plugin imports, usually bartorch.
```

You write a sequence plugin and a recon plugin. pulserver connects them to a
scanner through two services. The design calls run on the scanner host: they
answer each protocol edit, design and check the sequence, and store the design.
The reconstruction proxy runs on the reconstruction computer: it receives the
raw data of each series, completes it from the design that played it, and runs
your recon plugin. The two services share one thing, the design store.

Lesson 1 of the {doc}`Course <../examples/course>` runs one acquisition through
both services on the virtual scanner. This page describes what each service
does and what holds the pieces together.

## What pulserver does

:::{container} capabilities

- **Answers the interpreter's design calls, one command per call or through a warm server.** A warm server keeps the design engine loaded; a crashing plugin fails only its own call.

  Code: `pulserver design`, {mod}`pulserver.host`. Tests: *a forwarded call replies as the call answered in its own process*; *a crashing plugin fails only its call*; *a call is answered while another runs* (`test_design_server.py`).
- **Identifies a design by the SHA-256 of what it depends on.** The hash covers the plugin, its source and the sequence function's source, the package versions, the limits and check limits with the VOP file's contents, and the resolved protocol.

  Code: {func}`~pulserver.host.design_identity`. Tests: *one resolved protocol is one design*; *an edited plugin is another design of the same protocol*; *a design is one of the VOP file contents its SAR ratios came from* (`test_design_service.py`).
- **Designs a request once, whether it arrives as typed or as resolved.** A preset and the time it resolves to identify the same design.

  Code: {func}`~pulserver.host.design_identity`. Test: *a request is generated once whether or not it resolves to itself* (`test_design_service.py`).
- **Identifies an imported Pulseq chain by its files and the prescription it is imported at.**

  Code: `pulserver design import`. Test: *importing the same files at the same prescription is one design* (`test_design_service.py`).
- **Gives each design a 72-bit identifier that travels in three float32 fields of the raw-data header.** An identifier that already holds another identity is refused.

  Code: {func}`~pulserver.host.design_id`. Tests: *a design identifier is three integers a float32 holds exactly*; *a design identifier holding another identity is refused* (`test_design_service.py`).
- **Writes a design whole or not at all, and never modifies it.** Two processes designing the same protocol at once store it once.

  Code: {mod}`pulserver.host`. Tests: *a design being written is invisible until it is whole*; *two processes designing one protocol at once store it once*; *a stored design holds the files its manifest records and no other* (`test_design_service.py`).
- **Prunes the store least recently used first.** A design is used when it is written or found again, and one used within the day is kept.

  Code: `pulserver design prune`. Tests: *pruning removes the designs unused for longest first*; *a design found again is used again*; *pruning keeps a design used within the day whatever the limits* (`test_design_service.py`).
- **Pushes each design to a proxy on another computer before replying its identifier.** The proxy's intake stores a design only when every file matches the manifest, and a design that cannot be pushed fails the call.

  Code: {class}`~pulserver.proxy.DesignIntake`. Tests: *a generation pushes the design it replies*; *the intake refuses a tampered bundle with the reason* (`test_design_push.py`).
- **Reads each design once on the reconstruction side and keeps the most recently read.**

  Code: {class}`~pulserver.proxy.DesignCache`. Tests: *the least recently read design is read again when next named*; *a series naming no stored design is refused* (`test_proxy.py`).
- **Writes a log of each call and each reconstruction, including the output of the processes it spawns.**

  Code: {mod}`pulserver.host`. Tests: *a call naming a log file appends its outcome there* (`test_design_server.py`); *a logfile collects log output warnings and spawned children* (`test_logs.py`).

:::

## What the scanner and the engines do

- **The interpreter** is the vendor-specific program on the scanner. It shows
  the protocol in the scanner UI, loads the IR cache, plays the sequence and
  streams the raw data. It contains no sequence or reconstruction logic.
- **pypulseqpp** designs the sequence and writes the Pulseq files. Any tool
  that writes Pulseq can supply an imported chain instead.
- **The reconstruction engine** is whatever your recon plugin imports, usually
  bartorch. pulserver implements no reconstruction algorithm of its own.

## How it works

```{figure} ../_static/architecture.svg
:figclass: only-light

The two pulserver services between the scanner and the engines, and the
design store: one directory both reach, or the proxy's own, to which the design
calls push each design.
```

```{figure} ../_static/architecture-dark.svg
:figclass: only-dark

The two pulserver services between the scanner and the engines, and the
design store: one directory both reach, or the proxy's own, to which the design
calls push each design.
```

### One acquisition

1. On each protocol edit, the interpreter sends the plugin name, the scanner
   limits and the requested protocol. The design call evaluates the protocol
   with your plugin and replies with the values the design achieves and the
   scan time, or with the error your plugin raised ({doc}`protocol`).
2. When the scan is prepared, the call designs the whole sequence, checks it
   ({doc}`safety-checks`), converts it into the IR cache
   ({doc}`scanner-representation`), stores both, and replies with the design's
   identifier. The interpreter loads the cache and plays it.
3. The reconstruction client streams the raw data as MRD. The header names the
   design, and the client's configuration names the recon plugin.
4. The proxy completes the header and every readout from the design's Pulseq
   files and runs your plugin in a worker process ({doc}`reconstruction`). The
   images return to the console over the client's connection.

### Design identity

A design is identified by what it depends on, so no design call keeps state
between calls; the store is all that persists. A changed plugin, sequence
function or package version gives a new design for an unchanged protocol, and
so does a VOP file rewritten at the same path. The hash is taken over the
resolved protocol, and a design is made from its resolved protocol, so the
files of a design are a function of its identity.

The identifier is the first 18 hexadecimal digits of the identity. The
interpreter passes numbers to the reconstruction client in float32 fields of
the raw-data header, and a float32 holds every integer below $2^{24}$ exactly,
so the identifier takes three fields of six digits each.

### The store

```text
designs/
  3f9c0a1b2c3d4e5f60/
    sequence.seq          first file of the NextSequence chain, binary Pulseq
    sequence_main.seq     the remaining files, when there are prescans
    sequence.pseg         IR cache
    resolved.protocol
    manifest.json         identity and identifier, limits, package versions,
                          creation time, plugin and source, scan time,
                          field-of-view offset, SHA-256 of every file
```

A design is written into a staging directory and renamed into place once it is
complete. The reconstruction side reads the Pulseq files, not the IR cache: the
cache is derived for playout, and the Pulseq files remain the design of record.

When the two services run on different computers, the proxy either reads a
store both can reach, or keeps a store of its own that the design calls push to.

### What crosses each boundary

| Representation | Written by | Read by |
| --- | --- | --- |
| Protocol block (`[Protocol]` … `[Protocol End]`) | Design calls and interpreter | Both; the grammar is in {mod}`pulserver.protocol` and `pulseg_protocol.h` |
| Pulseq files, binary form | pypulseqpp, in a design call | The IR conversion and the reconstruction proxy |
| IR cache (`.pseg`) | {func}`pulserver.ir.convert` | The interpreter, through the C library in `src/c/` |
| MRD stream | Reconstruction client; the proxy, when forwarding | The proxy and its workers |

The orchestration is Python. The IR passes run only on the host and are C++,
in `src/cpp/ir/`. The library that reads the cache and walks it during playout
is linked into the interpreter, whose vendor toolchains accept ANSI C, so it is
C89, in `src/c/`.

## See it run

- {doc}`../generated/gallery/01-course/01_protocol_to_image`: one acquisition
  through both services on the virtual scanner.
- {doc}`../user-guide/running`: starting the two services, pushing designs and
  pruning the store.
- {doc}`../api/host`: the design calls and the store.
