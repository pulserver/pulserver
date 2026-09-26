# IR conversion

Segmentation of a Pulseq sequence and its `NextSequence` chain into the binary
IR cache the scanner interpreter loads beside the sequence file.

```{eval-rst}
.. currentmodule:: pulserver.ir
```

The chain is read with `pypulseqpp.Sequence`, text or binary. Event
deduplication, segmentation of the repetition pypulseqpp finds, and the
execution stream are computed in the compiled extension, and the cache is
written by the C library
the interpreter reads it back with. The files are read in the logical frame and
moved to the prescribed field-of-view offset before they are segmented. The
passes and the cache layout are described in {doc}`../explanations/ir-cache`.

| Object | Description |
| --- | --- |
| {obj}`~pulserver.ir.convert` | Segment a sequence file and write its IR cache beside it. |
| {obj}`~pulserver.ir.prescribe` | Move a logical-frame sequence to a prescribed field-of-view centre, in place. |
| {obj}`~pulserver.ir.check` | Timing, gradient, PNS and forbidden-band problems of a chain under a scanner's limits, in the physical frame. |
| {obj}`~pulserver.ir.CheckLimits` | The nerve model and forbidden bands a chain is checked against, and the VOPs of its SAR ratios, besides the gradient limits. |
| {obj}`~pulserver.ir.sar_ratios` | The RF energy of each subsequence of a chain at the VOPs, against the same repetitions of a hard, 180°, 1 ms reference pulse. |
| {obj}`~pulserver.ir.SarRatio` | The local and global SAR ratios of one subsequence, as the cache carries them. |
| {obj}`~pulserver.ir.summary` | Subsequences, segments, readouts, readout labels and RF spectral statistics of a sequence, from the chain or from its cache. |
| {obj}`~pulserver.ir.play` | Every block a cache plays, resolved as the scanner's playout resolves it, with its gradient waveforms on request. |
| {obj}`~pulserver.ir.plan_waves` | Where a playout holds a cache's waves in its waveform memory, all at once or a ring of slots per position, and whether it loads them in time: the layout the cache carries, or one for another budget. |
| {obj}`~pulserver.ir.WaveBudget` | The waveform memory, gradient raster, load rate and slots per position a playout affords the waves, which a cache is converted for. |
| {obj}`~pulserver.ir.sample_wave` | A cache's wave on a playout's gradient raster, as loaded into a region of its waveform memory. |
| {obj}`~pulserver.ir.playout` | Both stages of a segmented playout of a cache, the scan or its receive-gain prescan, recorded: what each position prepares, the registers of each block, what each wave reads from waveform memory and, on request, the waveforms each block plays. |
| {obj}`~pulserver.ir.Prescan` | The receive-gain calibration a playout plays instead of the scan: one subsequence, to the instance completing its readouts. |
| {obj}`~pulserver.ir.repetition_gradients` | The gradients of a subsequence's repetition of most gradient energy, as corner points and on a raster: what a scanner's heating and acoustic models are evaluated on. |
| {obj}`~pulserver.ir.chain` | Files of the `NextSequence` chain starting at a sequence file, in play order. |
| {obj}`~pulserver.ir.cache_path` | Cache file of a sequence file. |
