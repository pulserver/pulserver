# C++ API

C++17 over the C library: RAII types, value semantics and exceptions where
the C API returns codes into out-parameters, plus the sequence library and the
reconstruction-side reader that have no C counterpart. It **links** `src/c`
rather than restating it — one static library, one include path.

```{toctree}
:maxdepth: 1

pulseg
pulseq
recon
```

## The shape of it

`pulseg` is what an existing interpreter integrates: `Collection` — a parsed,
structured scan — `Opts`, and the safety entry points, with every C handle
owned and every error thrown. `pulseq` is the standalone sequence library:
`Sequence`, its events and files, k-space, moments, labels, deduplication and
expansion — the model PyPulseq keeps, held the way a million-block scan
needs it. `recon` is the other end: it reads a `.seq` chain into trajectories,
encoding spaces, labels and a sequence description, and enriches MRD
acquisitions with them.

## Which one do I want?

| I want to… | Use |
|---|---|
| add safety checking to an interpreter I already have | {doc}`pulseg`, and {doc}`../../examples/cpp/safety_only` |
| walk a scan's blocks and waveforms | {doc}`pulseg` |
| read, build, analyse or write a `.seq` | {doc}`pulseq` |
| feed a reconstruction service from the sequence file | {doc}`recon` |

The design side is on the hot path for million-block sequences, so an
allocation per block is measured before it is added; the reconstruction side
uses the standard library freely.
