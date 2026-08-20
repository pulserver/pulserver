# C API

What a scanner links against. ANSI C89, no dependencies, no allocator of its
own beyond `PULSEG_ALLOC` — the constraints an interpreter running on a
32-bit embedded target with an old toolchain imposes, taken as the design
brief rather than worked around.

```{toctree}
:maxdepth: 1

pulseg
safety
protocol
pulseq
```

## The shape of it

`pulseg` is the library an interpreter integrates: it reads a `.seq` chain
into a **collection** — the parsed, structured scan — and answers questions
about it. `safety` is the gate that collection has to pass before the scanner
plays it: gradient limits, continuity, acoustic resonance and PNS, with the
nerve model injected by the caller so no vendor formula lives here.
`protocol` is the other direction: the parameters a console UI shows, and the
bridge that asks a design host to build a sequence from them. `pulseq` is the
standalone `.seq` reader underneath all of it, which has no `pulseg`
dependency and can be linked on its own.

## Conventions

These hold across every header, so a signature can be read without looking
anything up.

| | |
|---|---|
| Prefix | every entry point is `pulseg_` or `pulseq_`, and `extern "C"` under C++ |
| Result | `PULSEG_SUCCESS` or a negative `PULSEG_ERR_*`, tested with `PULSEG_SUCCEEDED()` / `PULSEG_FAILED()` — and the `PULSEQ_` spellings in `pulseq` |
| Argument order | self, outputs, inputs — a pointer and the capacity after it count as one argument |
| Units | in the name: `_us`, `_hz`, `_hz_per_m`, `_hz_per_m_per_s` |
| Time | integer microseconds; a count that can pass two billion is a `double` |

The public API is exactly the contents of `src/c/include/`. Anything else
under `src/c/` — `pulseg_internal.h` above all — is private and changes
without notice.

## Where to start

An interpreter that already plays Pulseq and wants only the safety verdict
reads {doc}`safety` and the worked example in
{doc}`../../examples/cpp/safety_only`. One that wants the whole playout —
blocks, waveforms, chunking — starts at {doc}`pulseg`.
