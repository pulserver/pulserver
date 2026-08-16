# Gradient amplitude and slew

The simplest of the three safety gates, and the one every other check leans
on: no gradient sample may exceed the hardware's peak amplitude or internal
slew rate, and no transition between blocks may exceed the raster-step slew
limit either. It runs from the same PulSeg definitions and instances that
will be executed, on the host, before a scan is cached.

## The naive algorithm

Expand every block to its dense per-sample waveform, differentiate for slew,
and compare every sample against the limit. Correct, and unnecessary: a
trapezoid's extrema are its vertices, and an arbitrary waveform's are among
its raw samples — nothing about "is any sample too large" needs the samples
in between to be materialised.

## What Pulserver computes instead

**Amplitude and internal slew are cached per definition, not per instance.**
When a gradient definition is built, its normalised peak amplitude and
maximum internal slew rate are memoised once (`grad_definitions[...].amplitude`
/ `.slew_rate[shot_index]` in `csrc/src/structure/pulseg_structure.c`, one
entry per multishot variant). `check_max_grad` and `check_max_slew`
(`csrc/src/safety/pulseg_safety.c`) then walk `block_table`, and for every
block multiply the cached per-definition value by that block's per-instance
signed amplitude — a table lookup and a multiply, not a waveform expansion,
for every block in the collection.

**The limit is a vector norm, not three independent channels.** Both checks
accumulate $G_x^2+G_y^2+G_z^2$ (or the slew equivalent) per block and compare
against $3\times(\text{per-axis derated limit})^2$ — the per-axis limit is
pre-derated by $\sqrt3$ upstream so that, under an arbitrary rotation, no
single physical axis can exceed the scanner's true hardware peak. Evaluating
the *unrotated* waveform's vector magnitude against the physical limit is
therefore equivalent to checking every possible rotation at once, without
enumerating any of them.

**Boundary continuity is a separate, cheap dry-run.** `check_grad_continuity`
walks the execution stream once, comparing the cached **last sample** of one
block's definition against the cached **first sample** of the next, after
applying each instance's amplitude and rotation, and against the
subsequence's leading/trailing transition to zero. A waveform can be
individually legal — every sample under the peak, every internal step under
the slew limit — and still be rejected here, because joining two legal
waveforms at their shared boundary is a separate constraint from either
waveform being legal on its own.

Together, the three checks make the full-collection gate a walk over compact
per-block and per-definition tables: cost follows the number of blocks and
unique definitions, not the duration of a scan or the density of the
underlying raster. Timing is reported in {doc}`../benchmarks`.
