# Architecture

```{admonition} TL;DR
:class: tldr

- Two services: design calls on the scanner host, the reconstruction proxy on the reconstruction computer.
- Each design is stored once, under the SHA-256 of what it depends on, and never modified.
- The interpreter only presents, plays and streams; pypulseqpp designs and your plugin's engine reconstructs.
```

You write a sequence plugin and a recon plugin. The design calls answer each
protocol edit, design and check the sequence, and store it. The proxy receives
each series, completes it from the stored design and runs your recon plugin.
Lesson 1 of the {doc}`Course <../examples/course>` runs one acquisition through
both on the virtual scanner.

## What pulserver does

:::{container} capabilities

- **Answers the interpreter's design calls, one command per call or through a warm server.**

  Code: `pulserver design`, {mod}`pulserver.host`. Tests: *a forwarded call replies as the call answered in its own process*; *a crashing plugin fails only its call*; *a call is answered while another runs* (`test_design_server.py`).
- **Identifies a design by the SHA-256 of its plugin, sources, package versions, limits, VOP file and resolved protocol.**

  Code: {func}`~pulserver.host.design_identity`. Tests: *one resolved protocol is one design*; *an edited plugin is another design of the same protocol*; *a design is one of the VOP file contents its SAR ratios came from* (`test_design_service.py`).
- **Designs a request once, whether it arrives as typed or as resolved.**

  Code: {func}`~pulserver.host.design_identity`. Test: *a request is generated once whether or not it resolves to itself* (`test_design_service.py`).
- **Identifies an imported Pulseq chain by its files and the prescription it is imported at.**

  Code: `pulserver design import`. Test: *importing the same files at the same prescription is one design* (`test_design_service.py`).
- **Gives each design a 72-bit identifier that travels in three float32 fields of the raw-data header.**

  Code: {func}`~pulserver.host.design_id`. Tests: *a design identifier is three integers a float32 holds exactly*; *a design identifier holding another identity is refused* (`test_design_service.py`).
- **Writes a design whole or not at all, once, and never modifies it.**

  Code: {mod}`pulserver.host`. Tests: *a design being written is invisible until it is whole*; *two processes designing one protocol at once store it once*; *a stored design holds the files its manifest records and no other* (`test_design_service.py`).
- **Prunes the store least recently used first.**

  Code: `pulserver design prune`. Tests: *pruning removes the designs unused for longest first*; *a design found again is used again*; *pruning keeps a design used within the day whatever the limits* (`test_design_service.py`).
- **Pushes each design to a proxy on another computer, which stores it only if every file matches the manifest.**

  Code: {class}`~pulserver.proxy.DesignIntake`. Tests: *a generation pushes the design it replies*; *the intake refuses a tampered bundle with the reason* (`test_design_push.py`).
- **Reads each design once on the reconstruction side and keeps the most recently read.**

  Code: {class}`~pulserver.proxy.DesignCache`. Tests: *the least recently read design is read again when next named*; *a series naming no stored design is refused* (`test_proxy.py`).
- **Writes a log of each call and each reconstruction, including the output of the processes it spawns.**

  Code: {mod}`pulserver.host`. Tests: *a call naming a log file appends its outcome there* (`test_design_server.py`); *a logfile collects log output warnings and spawned children* (`test_logs.py`).

:::

## What the scanner and the engines do

- **The interpreter** shows the protocol, plays the IR cache and streams the raw data. It holds no sequence or reconstruction logic.
- **pypulseqpp** designs the sequence; any Pulseq tool can supply an imported chain instead.
- **The reconstruction engine** is whatever your recon plugin imports, usually bartorch.

## How it works

```{figure} ../_static/architecture.svg
:figclass: only-light

The two services between the scanner and the engines, and the design store they share.
```

```{figure} ../_static/architecture-dark.svg
:figclass: only-dark

The two services between the scanner and the engines, and the design store they share.
```

### One acquisition

1. Each protocol edit: the call evaluates it with your plugin and replies with the achieved values and scan time ({doc}`protocol`).
2. Scan preparation: the call designs, checks ({doc}`safety-checks`), converts ({doc}`scanner-representation`) and stores, and replies with the identifier.
3. Acquisition: the client streams MRD; the header names the design and the configuration names the recon plugin.
4. Reconstruction: the proxy completes the stream from the design ({doc}`raw-data`) and runs your plugin in a worker ({doc}`reconstruction`).

### Design identity

No design call keeps state; the store is all that persists. A changed plugin,
sequence function, package version or VOP file is a new design of the same
protocol. The identifier is the first 18 hexadecimal digits of the identity,
split into three float32 header fields because a float32 holds every integer
below $2^{24}$ exactly.

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

The proxy reads the Pulseq files, not the cache: the cache is derived for
playout, and the Pulseq files are the design of record.

### What crosses each boundary

| Representation | Written by | Read by |
| --- | --- | --- |
| Protocol block (`[Protocol]` … `[Protocol End]`) | Design calls and interpreter | Both; the grammar is in {mod}`pulserver.protocol` and `pulseg_protocol.h` |
| Pulseq files, binary form | pypulseqpp, in a design call | The IR conversion and the reconstruction proxy |
| IR cache (`.pseg`) | {func}`pulserver.ir.convert` | The interpreter, through the C library in `src/c/` |
| MRD stream | Reconstruction client; the proxy, when forwarding | The proxy and its workers |

Orchestration is Python, the IR passes are C++ (`src/cpp/ir/`), and the library
the interpreter links is C89 (`src/c/`), for the vendor toolchains.

## See it run

- {doc}`../generated/gallery/01-course/01_protocol_to_image`: one acquisition
  through both services on the virtual scanner.
- {doc}`../user-guide/running`: starting the two services, pushing designs and
  pruning the store.
- {doc}`../api/host`: the design calls and the store.
