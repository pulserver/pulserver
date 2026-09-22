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
directory they share.
```

The host daemon, {mod}`pulserver.host`, runs on the scanner host and answers
the PSD host process of each running sequence. The reconstruction proxy,
{mod}`pulserver.vre`, runs on the reconstruction computer and answers the
scanner's reconstruction client. One acquisition proceeds as follows.

1. The PSD host process opens a design session, naming the scanner-sequence
   plugin and the scanner limits.
2. On each protocol edit, the process requests validation. The host daemon
   designs the sequence with pypulseqpp and replies with the protocol the
   design achieves and the scan time, or with the error the design raised
   ({doc}`protocol`).
3. When the scan is prepared, the daemon generates the design: the Pulseq
   files and the IR cache beside them, stored as a revision of the session
   ({doc}`sessions`, {doc}`ir-cache`). The interpreter loads the IR cache and
   plays the sequence.
4. The reconstruction client streams the raw data of the series as MRD. The
   header names the session and revision the series was played from.
5. The proxy reads the revision's sequence files, completes the MRD header and
   every acquisition from them, and runs the reconstruction plugin the sequence
   names in a worker process ({doc}`reconstruction`). The images return to the
   console through the client's connection.

## Representations at each boundary

| Representation | Written by | Read by |
| --- | --- | --- |
| Protocol block (`[NimPulseqGUI Protocol]`) | Host daemon and interpreter | Both; the grammar is in {mod}`pulserver.protocol` and `pulseg_protocol.h` |
| Pulseq files, binary form | pypulseqpp, in the host daemon | The IR conversion and the reconstruction proxy |
| IR cache (`.pseg`; `.pge` on GE) | {func}`pulserver.ir.convert` | The interpreter, through the C library in `src/c/` |
| MRD stream | Reconstruction client | Reconstruction proxy and its workers |

The reconstruction side reads the Pulseq files of a revision, not its IR cache:
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
