# C API

What a scanner links against. ANSI C89, no dependencies, no allocator of its
own beyond `PULSEG_ALLOC` — the constraints an interpreter running on a
32-bit embedded target with an old toolchain imposes, taken as the design
brief rather than worked around.

```{toctree}
:maxdepth: 1

pulseq
types
protocol
file
checks
cache
generation
playout
```

````{only} not doxygen
```{note}
The function and struct reference on these pages is generated from the
headers by Doxygen, which is not installed in this build. The prose is
unaffected; `apt install doxygen` (or the equivalent) and rebuild to see it.
```
````

## Two packages

`pulseq` parses a `.seq` into the raw Pulseq model: blocks, the event
libraries they index, shapes, definitions. It has no `pulseg` dependency and
can be linked alone. It is a separate package because the file format is one
sequence-design language among the others this framework could read.

`pulseg` turns that into a **collection** — the deduplicated, segmented scan —
checks it against the hardware, and answers what an interpreter needs to play
it. `pulseg_convert_collection` is the seam; `pulseg_read` is the two
composed.

`pulseg` includes `pulseq`, never the reverse.

## Where each call belongs in an integration

Not every platform has all five as distinct entry points and the names differ;
the order does not.

| Stage | What happens | Pages |
|---|---|---|
| **Setup** | Declare the scanner's limits, open a connection to the design host, ask it for the default protocol, and drive the console UI from it. | {doc}`types`, {doc}`protocol` |
| **Prescription check** | Every time the operator changes a parameter: ask the design host whether the protocol is still playable, and read the scan time off the file without parsing it. | {doc}`protocol`, {doc}`file` |
| **Preparation** | Ask for the `.seq`, read it, convert it to a collection, gate it against the limits, and serialise the cache the later stages read. | {doc}`pulseq`, {doc}`file`, {doc}`checks`, {doc}`cache` |
| **Waveform generation** | Load the generation-stage cache, materialise the distinct waveforms, and plan how they fit in waveform memory. | {doc}`cache`, {doc}`generation` |
| **Playout** | Load the scan-stage cache and walk the execution stream, one block instance at a time, in the order it plays. | {doc}`cache`, {doc}`playout` |

{doc}`../../examples/c/index` has one worked example per stage.

## By role

For the calls that serve more than one stage:

| | |
|---|---|
| **Reading a `.seq`** | The raw file model, on its own terms. {doc}`pulseq` |
| **Types and limits** | `pulseg_opts` and the rasters everything is judged against; the list and buffer types that carry variable-length arguments. {doc}`types` |
| **Errors and diagnostics** | The code, the message, the hint, and the block or axis that provoked it. {doc}`types` |
| **Structuring** | Turning parsed files into a collection, and asking what it says about itself. {doc}`file` |
| **Checks** | Rasters, gradient amplitude, slew, continuity, PNS and mechanical resonance — together or one at a time. {doc}`checks` |
| **Structure and events** | What the scan is made of, and the waveforms behind it. {doc}`generation` |
| **Traversal** | The cursor over the execution stream. {doc}`playout` |
| **Cache** | The binary sidecar, whole or by section. {doc}`cache` |

## Conventions

These hold across every header, so a signature can be read without looking
anything up.

| | |
|---|---|
| Prefix | every entry point is `pulseq_` or `pulseg_`, and `extern "C"` under C++ |
| Result | `PULSEQ_SUCCESS` / `PULSEG_SUCCESS`, or a negative `*_ERR_*`, tested with `PULSEG_SUCCEEDED()` / `PULSEG_FAILED()` and their `PULSEQ_` spellings |
| Argument order | self, `[out]`, `[in,out]`, `[in]` |
| Variable-length arguments | a count and its array travel as one type — `pulseg_forbidden_band_list`, `pulseg_text_buffer` — never as an interleaved pair |
| Many-argument calls | where the knobs outnumber the subject, they are gathered into a request struct: `pulseg_mech_resonances_request`, `pulseg_check_plan_config` |
| Units | in the name: `_us`, `_hz`, `_hz_per_m`, `_hz_per_m_per_s` |
| Time | integer microseconds; a count that can pass two billion is a `double` |
| Ownership | a getter that allocates says so and names its `_free`; anything else returns a borrowed view valid while the collection is |

The public API is exactly the contents of `src/c/include/`. Anything else
under `src/c/` — `pulseg_internal.h` above all — is private and changes
without notice.

## C++

{doc}`../cpp/index` is the same library with C++17 types. Its pages mirror
these one for one.
