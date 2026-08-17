# Python API

Designing a sequence, describing it to the console, writing it, checking it,
and reconstructing what comes back.

```{toctree}
:maxdepth: 1

design
app_sequence
pypulseq
protocol
io
recon
app_recon
```

## The shape of it

`design` holds the modules a sequence is assembled from — an excitation, a
readout, a preparation — each of which solves its own timing. `app.sequence` holds
complete sequences built from them, which double as the worked examples and
as the validation corpus. `pypulseq` is the sequence object itself: blocks,
events, files, and the analysis methods (k-space, labels, PNS, spectra) that
read a built sequence back. `protocol` declares what the scanner UI shows.
`recon` and `app.recon` are the other end: physics operators, models and
optimizers, and the ready-made reconstructions paired with the zoo.
