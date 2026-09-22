# Architecture

An acquisition through pulserver involves four parties:

| Party | Runs on | Role |
| --- | --- | --- |
| Scanner interpreter | Scanner | Vendor-specific playout: presents the protocol in the scanner UI, loads the design, plays it, and streams the raw data |
| Host daemon, `pulserver.host` | Scanner host | Resolves the protocol an operator edits and writes the design the interpreter loads |
| Reconstruction proxy, `pulserver.vre` | Reconstruction computer | Receives each series, attaches what the sequence states about its readouts, and runs a reconstruction |
| Engines | Host daemon and proxy workers | pypulseqpp designs the sequence; bartorch, or whatever a plugin imports, reconstructs it |

The interpreter carries no sequence or reconstruction logic: it calls
pulserver's services and plays what they return. Pulserver in turn implements
neither sequence design nor reconstruction algorithms; code that belongs to an
engine is contributed to that engine.

## From protocol to images

1. A PSD host process on the scanner opens a design session with the host
   daemon, naming the scanner-sequence plugin and the scanner limits.
2. Each time the operator edits the protocol, the process asks the daemon to
   validate it. The daemon designs the sequence with pypulseqpp and replies
   with the protocol the sequence will play (the shortest echo time, a
   bandwidth on the sampling raster) and the scan time, or with the error the
   design raised.
3. When the scan is prepared, the daemon generates the design: the Pulseq
   files and the IR cache beside them, stored as a revision of the session.
   The interpreter loads the cache and plays the sequence.
4. The scanner's reconstruction client streams the raw data as MRD to the
   reconstruction proxy. The header names the session and revision the series
   was played from.
5. The proxy reads that revision's sequence files, fills in the MRD header and
   every acquisition from them, and hands the stream to a worker running the
   reconstruction plugin the sequence names.
6. The worker's images return to the console through the same connection.

## Where each representation lives

| Representation | Written by | Read by |
| --- | --- | --- |
| Protocol block (`[NimPulseqGUI Protocol]`) | Host daemon and interpreter | Both; grammar in {mod}`pulserver.protocol` and `pulseg_protocol.h` |
| Pulseq files (`.seq`, binary form) | pypulseqpp, in the host daemon | The IR conversion and the reconstruction proxy |
| IR cache (`.pseg`, `.pge` on GE) | `pulserver.ir` | The interpreter, through the C library in `src/c/` |
| MRD stream | Reconstruction client | Reconstruction proxy and its workers |

The Pulseq file is the design of record. The IR cache is derived from it for
playout and never read by the reconstruction side, which reads the Pulseq files
of the revision instead.

## Language boundaries

The Python package holds the orchestration. The segmentation of a sequence into
the IR runs only on the host and is C++ (`src/cpp/ir/`). The cache reader and
the accessors a playout walks it with are linked into the interpreter, so they
are ANSI C (C89) in `src/c/`, with no dependency on anything the host runs; the
build compiles them with `-std=c90 -pedantic-errors`, as a scanner build does.
