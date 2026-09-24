# IR conversion

Segmentation of a Pulseq sequence and its `NextSequence` chain into the binary
IR cache the scanner interpreter loads beside the sequence file.

```{eval-rst}
.. currentmodule:: pulserver.ir
```

The chain is read with `pypulseqpp.Sequence`, text or binary. Event
deduplication, repetition detection, segmentation and the execution stream are
computed in the compiled extension, and the cache is written by the C library
the interpreter reads it back with. The files are read in the logical frame and
moved to the prescribed field-of-view offset before they are segmented. The
passes and the cache layout are described in {doc}`../explanations/ir-cache`.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.ir.convert` | Segment a sequence file and write its IR cache beside it. |
| {obj}`~pulserver.ir.prescribe` | Move a logical-frame sequence to a prescribed field-of-view centre, in place. |
| {obj}`~pulserver.ir.check` | Timing, gradient, PNS, forbidden-band and VOP SAR problems of a chain under a scanner's limits, in the physical frame. |
| {obj}`~pulserver.ir.CheckLimits` | The nerve model, forbidden bands and VOPs a chain is checked against, besides the gradient limits. |
| {obj}`~pulserver.ir.summary` | Subsequences, segments, readouts and RF spectral statistics of a sequence, from the chain or from its cache. |
| {obj}`~pulserver.ir.play` | Every block a cache plays, resolved as the scanner's playout resolves it. |
| {obj}`~pulserver.ir.chain` | Files of the `NextSequence` chain starting at a sequence file, in play order. |
| {obj}`~pulserver.ir.cache_path` | Cache file of a sequence file. |
