# The structural TR, and segments

A Pulseq file is a flat list of blocks. Playing it is not: a scanner
executes *segments* — short reusable units it prepares once and replays with
per-instance amplitudes — and it gates the scan on quantities defined over a
*repetition*: SAR over the worst TR, gradient heating over the duty cycle,
acoustic response over the periodic drive. Neither the segment nor the
repetition is written in the file.

Pulserver derives both from the block content, and derives them the same way
on every file. This page explains what is derived, what it is used for, and
why the alternative — annotating the file — was rejected.

## The repeating unit is a property of the content

Take the first block, and ask whether the stream is periodic with period
$P$: whether block $n$ and block $n + P$ have the same normalized structure —
same duration, same events, same waveform *shapes* — for every $n$. The
smallest $P$ that holds over the whole table is the **structural TR**, and
`tr_size` blocks is its length.

Two properties make this well-posed:

- **Amplitudes are excluded.** A phase encode steps from line to line; a
  spoiler phase advances every shot; an RF pulse alternates sign. Those are
  per-instance parameters of one structure, not different structures. The
  comparison is on the normalized identity — shape by value, trapezoid by
  its four times, ADC by count, dwell and delay — which is exactly the
  identity Pulseq's own event libraries deduplicate on.
- **Pure delays match any pure delay.** A TI fill or a TR pad whose duration
  varies is one position whose duration is a runtime parameter, not a
  different sequence. This is the same relaxation the Pulseq specification
  makes for its reserved implicit-delay blocks.

For a gradient echo the answer is the obvious one: excitation, prewinder,
readout, rewinder, spoiler, pad — six blocks, repeated once per line. For an
FSE it is the whole echo train, because that is the unit that repeats. For a
ZTE it is one view: two blocks, the pulse and the acquisition, because a ZTE
really does repeat every 360 µs. Detection reports what the sequence is, not
what its author called it.

```{note}
`Sequence.declare_tr()` writes the detected block count into
`[DEFINITIONS] TRSize` when the file is written, so a consumer that wants the
answer need not re-derive it. It remains a *hint*: the interpreter verifies
the pattern repeats and falls back to full detection, and a reconstruction
that does not find it simply skips the description it would have built.
```

## What the TR is for

Three consumers, all of which would otherwise have to guess:

**Safety.** SAR is defined per unit time over the worst repetition; a scan
whose flip angle varies shot to shot has a worst one that is not the first
one. The RF checks walk the TR instances and take the worst B1rms, not the
average and not the first. The gradient-heating and acoustic checks need a
*period* to compute a spectrum over at all — see
{doc}`../safety/mechanical_resonance`.

**Playout.** A segment is prepared once and replayed; the boundary of the
repeating unit is where the interpreter can safely close a segment and know
the next instance starts in the same state.

**Reconstruction.** The sequence description a reconstruction reads — RF
definitions, echo positions, the ADC roles — is derived over one TR window
and applies to every instance of it. Deriving it over the whole scan would
be the same answer at a million times the cost.

## Segments are the TR, partitioned by structure

Within the TR, Pulserver partitions blocks into **segments**: maximal runs
that repeat identically across instances. A gradient echo yields one segment
(the whole TR) plus the pad; an EPI shot yields the fat saturation, the
train, and the frame-counter block, because those three do not repeat with
the same period as each other.

The partition satisfies the constraints the PulSeg IR specification places on
a virtual segment — every instance has the same block count and the same
normalized structure — and it is checked to: `tests/python/test_pulseg_oracle.py`
validates the interpreter's partition against an independent implementation
of the specification's rules, over the whole zoo.

### The declared divergence: content, not annotation

The PulSeg specification makes `TRID` annotation the segment-boundary
convention: the designer labels where segments begin. Pulserver does not read
it for that purpose, and this is a deliberate divergence rather than an
omission.

An annotation is a second source of truth. It can disagree with the content —
a designer edits the blocks and forgets the label — and when it does, the
interpreter is obliged to believe something the sequence does not do. Every
downstream guarantee (the same block count per instance, the same structure,
a segment that can be prepared once) is a property of the *content*, so
deriving the partition from the content makes those guarantees true by
construction instead of by trust.

`TRID` still has a job, a narrower one: when a hyper-TR interleaves several
subsequences, it tells the RF checks which acquisitions belong to which
constituent TR, so SAR and coil heating are computed per constituent rather
than conservatively over the whole. It is never load-bearing — absent it, the
conservative check runs and the hardware monitor still stands.

Concatenated subsequences need no label at all: they are separate `.seq`
files linked by `NextSequence`, loaded as a collection and evaluated
independently.

## What detection cannot do

Two honest limits, both visible in the zoo:

- **The period is anchored at the first block.** A scan whose first
  repetition differs from the rest — a catalyst before a bSSFP train — is
  read as one long TR rather than a prologue plus a period, and a
  single-slice train longer than 15 s (`SINGLE_TR_MAX_DURATION_US`) is
  refused rather than mis-segmented. Materializing the prologue with
  `Sequence.expand_repeats()` gives detection a uniform stream.
- **Granularity is a choice, not a fact.** Reading a scan as one segment
  played once breaks no rule; so does the finest partition. Pulserver picks
  the finest partition that satisfies the constraints, because that is the
  one that reuses the most on the scanner, but a different interpreter
  choosing differently is still conformant. The oracle checks conformance,
  not agreement.

## Seeing it

```python
from pulserver.seqzoo import gre_2d

seq = gre_2d.main(n_x=64, n_y=64, n_slices=3)

seq.declare_tr()              # 6 -- blocks in one repetition, also written
                              # into [DEFINITIONS] TRSize by write()
seq.plot(tr="worst_case")     # the repetition the safety checks used
seq.sequence_descriptor()     # what a reconstruction reads off one TR
```

Plotting the TR is usually the fastest way to see whether detection found
what you meant: `tr="worst_case"` draws the positional-maximum envelope the
gradient checks run on, and an integer index draws that actual repetition.
