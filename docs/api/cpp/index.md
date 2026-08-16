# C++ API

The C library with RAII types, plus the reconstruction-side reader. Header
only over `pulseg`/`pulseq`: one static library to link, one include path.

```{toctree}
:maxdepth: 1

pulseg
recon
```

- **`pulseg`** — `Collection` (a parsed, structured scan), `Opts` (the system
  it is checked against), and the safety entry points. This is what an
  existing interpreter integrates: see
  {doc}`../../examples/cpp/safety_only`.
- **`pulseq`** — the standalone sequence library: `Sequence`, its events and
  files, k-space, moments, labels, deduplication, expansion.
- **`recon`** — reads a `.seq` chain into trajectories, encoding spaces,
  labels and a sequence description, and enriches MRD acquisitions with them.
