# Architecture

A Pulseq sequence is a vendor-independent description of an acquisition. To
acquire data with one on a clinical scanner, four tasks have to be performed:
the sequence is designed for the protocol the operator prescribes, converted
into a form the scanner hardware can play, played, and its raw data
reconstructed into images returned to the console. Pulserver performs the
first, second and fourth of these on computers beside the scanner, and a
vendor-specific interpreter performs the third on the scanner itself.

## Layers

Four layers are kept distinct throughout the documentation.

Pulseq representation
: The content of a `.seq` file: blocks, event libraries, shapes, definitions
  and extensions. It is the design of record for every acquisition.

Engines
: [pypulseqpp](https://github.com/pulserver/pypulseqpp) designs and writes the
  sequence. The reconstruction is whatever a reconstruction plugin imports,
  normally [bartorch](https://github.com/mcencini/bartorch). Pulserver
  implements neither sequence design nor reconstruction algorithms.

Orchestration
: Pulserver: the resolution of a protocol against a sequence application, the
  storage of the resulting designs, the conversion of a design into the scanner
  IR, and the routing and enrichment of raw data.

Scanner execution
: The interpreter: the vendor-specific program that presents the protocol in
  the scanner UI, loads the IR, plays the sequence and streams the raw data. It
  contains no sequence or reconstruction logic.

## Services and data flow

```{figure} ../_static/architecture.svg
The two pulserver services between the scanner and the engines, and the
design store: one directory both reach, or the proxy's own, to which the design
calls push each design.
```

The design calls, {mod}`pulserver.host`, run on the scanner host and answer
the interpreter host process of each running sequence, one command per call or
through a warm server. The reconstruction proxy, {mod}`pulserver.vre`, runs on
the reconstruction computer and answers the scanner's reconstruction client.
One acquisition proceeds as follows.

1. On each protocol edit, the interpreter host process requests validation,
   naming the scanner-sequence plugin, the scanner limits and the requested
   protocol. The call constructs the application with pypulseqpp and replies
   with the protocol the design achieves and the scan time, or with the error
   the design raised ({doc}`protocol`).
2. When the scan is prepared, the process requests the design. The call designs
   the sequence, checks it, and stores the Pulseq files and the IR cache as a
   design of the design store, pushes the design to the proxy when the proxy
   keeps a store of its own, and replies its identifier ({doc}`designs`,
   {doc}`ir-cache`). The interpreter loads the IR cache and plays the sequence.
3. The reconstruction client streams the raw data of the series as MRD. The
   header names the design the series was played from.
4. The proxy reads the design's sequence files, completes the MRD header and
   every acquisition from them, and runs the reconstruction plugin the sequence
   names in a worker process, or forwards the completed series to a
   reconstruction server on another computer ({doc}`reconstruction`). The
   images return to the console through the client's connection.

## Representations at each boundary

| Representation | Written by | Read by |
| --- | --- | --- |
| Protocol block (`[NimPulseqGUI Protocol]`) | Design calls and interpreter | Both; the grammar is in {mod}`pulserver.protocol` and `pulseg_protocol.h` |
| Pulseq files, binary form | pypulseqpp, in a design call | The IR conversion and the reconstruction proxy |
| IR cache (`.pseg`) | {func}`pulserver.ir.convert` | The interpreter, through the C library in `src/c/` |
| MRD stream | Reconstruction client; the proxy, forwarding | Reconstruction proxy and its workers; a reconstruction server |

The reconstruction side reads the Pulseq files of a design, not its IR cache:
the cache is a derived representation for playout, and the Pulseq files remain
the design of record.

## Implementation languages

The orchestration is Python. The passes that compute the IR run only on the
host and are C++, in `src/cpp/ir/`. The code that reads the IR cache and walks
it during playout is linked into the interpreter, whose vendor toolchains
accept ANSI C; it is therefore C89, in `src/c/`, and depends on nothing that
runs only on the host.

## See also

* {doc}`../user-guide/running` — starting the two services.
* {doc}`../api/index` — the Python interface of each service.
