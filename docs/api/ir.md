# IR conversion

Segmentation of a Pulseq sequence and its `NextSequence` chain into the binary
IR cache the scanner interpreter loads beside the sequence file.

```{eval-rst}
.. currentmodule:: pulserver.ir
```

The chain is read with `pypulseqpp.Sequence`, text or binary. Event
deduplication, repetition detection, segmentation and the execution stream are
computed in the compiled extension, and the cache is written by the C library
the interpreter reads it back with. The passes and the cache layout are
described in {doc}`../explanations/ir-cache`.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.ir.convert` | Segment a sequence file and write its IR cache beside it. |
| {obj}`~pulserver.ir.check` | Timing and gradient problems of a chain under a scanner's limits. |
| {obj}`~pulserver.ir.summary` | Subsequences, segments, readouts and RF spectral statistics of a sequence, from the chain or from its cache. |
| {obj}`~pulserver.ir.chain` | Files of the `NextSequence` chain starting at a sequence file, in play order. |
| {obj}`~pulserver.ir.cache_path` | Cache file of a sequence file. |
