# Python API

Designing a sequence, describing it to the console, writing it, checking it,
and reconstructing what comes back.

```{toctree}
:maxdepth: 1

pypulseq
design
recon
mrd
apps
```

## The shape of it

`pypulseq` is the event layer: the `Sequence` object, the factories that build
blocks, the sampling and scheduling helpers a scan loop indexes, the analysis
methods (k-space, labels, PNS, spectra) that read a built sequence back, and
the file I/O either side of it. `design` is the toolbox above it — an
excitation, a preparation, one readout TR — each of which solves its own
timing and publishes the events the loop plays. `design` also carries the
scanner protocol: what a plugin is, and the typed parameters a console builds
its controls from. `recon` is the other end: buffers, physics operators,
solvers, priors and the plugin contract. `mrd` is the vocabulary the two ends
share — what a scanner sends, how a scan is described, the array operations
both sides read and write with, and the files a study is stored as. `apps`
holds the complete plugins built on all of them, the sequence zoo and the
reconstructions paired with it, together with the protocol declarations that
put them on a scanner UI.
