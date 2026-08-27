# C API

What a scanner links against. ANSI C89, no dependencies, no allocator of its
own beyond `PULSEG_ALLOC` — the constraints an interpreter running on a
32-bit embedded target with an old toolchain imposes, taken as the design
brief rather than worked around.

```{toctree}
:maxdepth: 1

types
protocol
file
checks
cache
generation
playout
pulseq
```

````{only} not doxygen
```{note}
The function and struct reference on these pages is generated from the
headers by Doxygen, which is not installed in this build. The prose is
unaffected; `apt install doxygen` (or the equivalent) and rebuild to see it.
```
````

## Two packages

`pulseg` is the interpreter library: it reads a `.seq` chain into a
**collection** — the parsed, structured scan — gates it against the hardware,
and answers what an interpreter needs to play it.

`pulseq` is the `.seq` reader underneath, with no `pulseg` dependency and its
own `pulseq_` prefix. It is a separate package because the file format is one
sequence-design language among the possible others this framework could read,
and a vendor may want only it.

## Where each call belongs in an integration

An interpreter reaches the library at a handful of points, and which page you
want depends on which one you are writing. The names below are vendor-neutral;
the GE hook each corresponds to is given once, as a worked reference.

| Stage | What happens | Pages |
|---|---|---|
| **Setup** <br/>(GE `CVInit`) | Declare the scanner's limits, open a connection to the design host, ask it for the default protocol, and drive the console UI from it. | {doc}`types`, {doc}`protocol` |
| **Prescription check** <br/>(GE `CVEval`) | Every time the operator changes a parameter: ask the design host whether the protocol is still playable, and read the scan time off the file without parsing it. | {doc}`protocol`, {doc}`file` |
| **Preparation** <br/>(GE `Predownload`) | Ask for the `.seq`, read it, convert it to a collection, gate it against the limits, and serialise the cache the later stages read. | {doc}`file`, {doc}`checks`, {doc}`cache` |
| **Waveform generation** <br/>(GE `Pulsegen`) | Load the generation-stage cache, materialise the distinct waveforms, and plan how they fit in waveform memory. | {doc}`cache`, {doc}`generation` |
| **Playout** <br/>(GE `Scan`) | Load the scan-stage cache and walk the execution stream, one block instance at a time, in the order it plays. | {doc}`cache`, {doc}`playout` |

## By role

Several calls serve more than one stage. Grouped by what they are rather than
by when they are used:

| | |
|---|---|
| **Types and limits** | `pulseg_opts` and the rasters everything is judged against; the list and buffer types that carry variable-length arguments. {doc}`types` |
| **Errors and diagnostics** | The code, the message, the hint, and the block or axis that provoked it. {doc}`types` |
| **Sequence file I/O** | Peeking, reading and converting a `.seq` chain. {doc}`file`, {doc}`pulseq` |
| **Checks** | Rasters, gradient amplitude, slew, continuity, PNS and mechanical resonance — together or one at a time. {doc}`checks` |
| **Structure and events** | What the scan is made of, and the waveforms behind it. {doc}`generation` |
| **Traversal** | The cursor over the execution stream. {doc}`playout` |
| **Cache** | The binary sidecar, whole or by section. {doc}`cache` |

## Conventions

These hold across every header, so a signature can be read without looking
anything up.

| | |
|---|---|
| Prefix | every entry point is `pulseg_` or `pulseq_`, and `extern "C"` under C++ |
| Result | `PULSEG_SUCCESS` or a negative `PULSEG_ERR_*`, tested with `PULSEG_SUCCEEDED()` / `PULSEG_FAILED()` — and the `PULSEQ_` spellings in `pulseq` |
| Argument order | self, `[out]`, `[in,out]`, `[in]` |
| Variable-length arguments | a count and its array travel as one type — `pulseg_forbidden_band_list`, `pulseg_text_buffer` — never as an interleaved pair |
| Many-argument calls | where the knobs outnumber the subject, they are gathered into a request struct: `pulseg_mech_resonances_request`, `pulseg_check_plan_config` |
| Units | in the name: `_us`, `_hz`, `_hz_per_m`, `_hz_per_m_per_s` |
| Time | integer microseconds; a count that can pass two billion is a `double` |
| Ownership | a getter that allocates says so and names its `_free`; anything else returns a borrowed view valid while the collection is |

The public API is exactly the contents of `src/c/include/`. Anything else
under `src/c/` — `pulseg_internal.h` above all — is private and changes
without notice.

## Where to start

An interpreter that already plays Pulseq and wants only a safety verdict reads
{doc}`checks`. One that wants the whole playout starts at {doc}`file` and
follows the stage table down.
