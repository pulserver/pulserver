# The structural TR, and segments

```{admonition} TL;DR
:class: tip

- The **structural TR** is the smallest block period $P$ over which the
  normalised structure repeats. Everything a playout can vary — amplitudes,
  offsets, rotations, which readout a block digitises with — is excluded from
  the comparison, and a pure delay matches any pure delay.
- A file may declare its own period with `[DEFINITIONS] TRSize`. It is
  **verified** before it is honoured, and discarded if the blocks contradict it.
- **Segments** are the TR partitioned by structure: maximal runs that repeat
  identically across instances. Boundaries go at the last legal seam before each
  excitation, and a seam is legal only where **every** gradient axis rests at
  zero on both sides.
- Both are **derived from the block content**, never from a `TRID` annotation:
  an annotation is a second source of truth that a later edit can contradict.
- The TR is the *window* every safety check runs over. What is put in that
  window is a separate question — see {doc}`../safety/canonical_tr`.
```

A Pulseq file is a flat list of blocks. Playing it is not: a scanner executes
*segments* — short reusable units it prepares once and replays with per-instance
amplitudes — and it gates the scan on quantities defined over a *repetition*.
Neither the segment nor the repetition is written in the file.

## Detecting the repeating unit

Take the first block, and ask whether the stream is periodic with period
$P$: whether block $n$ and block $n + P$ have the same normalized structure —
the same duration, playing on the same channels — for every $n$. The smallest
$P$ that holds over the whole table is the **structural TR**, and `tr_size`
blocks is its length.

```{figure} ../assets/segments/tr_and_segments_schematic.png
The block stream, the repeating unit found in it, and the segments that unit is
cut into — drawn on an inversion-prepared gradient echo at a train length of
two. A spoiled gradient echo would not show it: its TR *is* one shot, so every
segment in it is played once. Here the readout segment is replayed inside the
TR, and one delay segment serves both the TI fill and the recovery pad.
```

Two properties make this well-posed:

**Everything a playout can vary is excluded.** A phase encode steps, a spoiler's
phase advances, an RF pulse alternates sign, a radial view rotates, a navigator
acquires on some repetitions and not others. Those are per-instance parameters
of one structure, so the comparison ignores amplitudes, which waveform variant a
gradient plays, the frequency and phase offsets of RF and ADC, the rotation,
whether a trigger fires, and whether the block's ADC acquires this time round.
What is compared is the timing each event occupies inside the block: an RF pulse
by its delay and its magnitude, phase and time shapes; a trapezoid by its delay,
rise, flat and fall; an arbitrary gradient by its delay and its time shape, or
its sample count where the samples sit on the raster. The ADC is left out
altogether — which readout a block digitises with is carried by the instance, so
a train alternating between two of them still repeats every shot rather than
every pair.

**Pure delays match any pure delay.** A TI fill or a TR pad whose duration
varies is one position whose duration is a runtime parameter, not a different
sequence — the same relaxation the Pulseq specification makes for its reserved
implicit-delay blocks.

For a gradient echo the answer is the obvious one: five blocks, repeated once
per line. For an FSE it is the whole echo train. For a ZTE it is the whole shell
— the ramp onto the first spoke, then a pulse and an acquisition per view —
because the shell opens and closes differently from the way it runs. Detection
reports what the sequence is, not what its author called it.

## Declaring the period instead of deriving it

A file may state its own period. `[DEFINITIONS] TRSize` carries a block
count, and the interpreter reads it before it searches: it checks that the
block table really does repeat at that period — the same test it would apply
to a period of its own finding — and takes it when it holds.

A declaration the blocks contradict is discarded and detection runs as though it
had never been made, so a stale number costs one verification pass and misleads
nothing.

What a declaration is *for* is the case detection cannot reach. Detection always
returns the shortest period. A designer who wants a longer window — a hyper-TR
spanning two interleaved subsequences, so the RF checks see the pair rather than
one of them — states it and gets it, provided the blocks repeat at it. Every
consumer treats a longer window conservatively: fewer TRs, a bigger safety
window, less segment reuse.

`Sequence.declare_tr()` writes the detected count, and `write()` and
`write_binary()` call it, so every written file carries the declaration
whether or not anyone asked for one. A reconstruction derives its sequence
description only when the definition is present.

## What the TR is used for

Three consumers, all of which would otherwise have to guess:

**Safety.** Every check is defined over a repetition: SAR per unit time over
the worst one, gradient heating over a duty cycle, the acoustic response over a
drive that has to be periodic before it can be a spectrum at all. The TR is the
window they all run over. *Which* repetition each check evaluates in that
window, and how it is chosen without evaluating them all, is
{doc}`../safety/canonical_tr`.

**Playout.** A segment is prepared once and replayed; the boundary of the
repeating unit is where the interpreter can close a segment and know the next
instance starts in the same state.

**Reconstruction.** The sequence description — RF definitions, echo positions,
ADC roles — is derived over one TR window and applies to every instance.

## Segmentation: the TR partitioned by structure

Within the TR, Pulserver partitions blocks into **segments**: maximal runs
that repeat identically across instances. A gradient echo yields one segment
(the whole TR) plus the pad; an EPI shot yields the fat saturation, the
train, and the frame-counter block, because those three do not repeat with
the same period as each other.

The partition satisfies the constraints the PulSeg specification places on a
virtual segment: every instance has the same block count and the same
normalized structure.

A segment also carries the readout, not only the timing — see
{ref}`shots that acquire nothing <acquire-nothing>` below.

### Where a boundary is allowed

A boundary is a place where the sequencer stops one prepared unit and starts
the next. The gradient cannot be live across such a stop, so the rule is a
statement about the waveforms and not about the events:

```{figure} ../assets/segments/segmentation_rules.png
The seams of one TR, read across all three gradient axes at once. A seam is
legal only where every axis rests at zero on both sides of it; one live axis
is enough to rule it out, whatever the other two are doing. The cuts are then
placed at the last legal seam before each excitation.
```

**A seam is legal when both sides rest at zero — on every axis at once.** The
test is applied to $G_x$, $G_y$ and $G_z$ separately and all three have to
pass: the previous block's last sample and the next block's first, each within
`SEG_ZERO_GRAD_THRESHOLD_HZ_PER_M` (100 Hz/m) of zero. The threshold is an
absolute amplitude, not a slew allowance; nothing about how fast the waveform
was moving enters into it, because what has to be true at a boundary is that
the sequencer can stop there, and a gradient at zero is one it can stop under.

One axis is enough to disqualify a seam. A readout carrying its gradient into
the rewinder that follows it has no legal seam between the two, and that is
visible on the readout axis. A slice-select running straight into its rephaser
has no legal seam either — and there the two in-plane axes are both flat across
it, so the seam looks legal on the axis a single-axis sequence diagram would
draw. The rule is a statement about the gradient vector, not about any one
channel.

**The whole pattern must start and end at zero.** If the first block of the
TR begins under a gradient, or the last one ends under one, the sequence is
refused rather than segmented — the TR is played back to back with copies of
itself, so its two ends are a seam like any other.

**A change of rotation state forces a boundary.** The rotation matrix is
programmed once per segment, at its start, so `NOROT` and PMC are part of a
segment's identity. Where either changes, a boundary is placed; if that
position is not a legal seam, the sequence is refused, because a rotation
remaps the axes and cannot take effect under a live waveform. Absorbing it —
letting the block play in its predecessor's frame — is silently the wrong
image.

### Where a boundary is placed

Among the legal seams, the cuts go **immediately before the RF**: for each
excitation, the last legal seam preceding it closes the segment that was
open. The leading run up to the first acquisition is closed the same way, at
the last legal seam before the RF that precedes that acquisition.

That makes a segment a *shot*: it begins with its excitation and ends where the
next is about to start, which is the unit the scanner prepares and the unit a
per-instance amplitude table indexes.

Where an RF pulse arrives with no legal seam since the last cut — a
continuous-gradient family whose waveform never returns to zero between pulses —
cutting stops for the rest of the pattern and what remains becomes one segment.
Nothing is refused; the sequence simply has less reuse in it than a Cartesian
one.

### Two further splits

**Pure delays are split off.** A run of pure-delay blocks at the head or the
tail of a segment becomes one segment per block. A delay is one position whose
duration the playout sets, so keeping it inside a segment would bind a
duration into a prepared unit that could otherwise be reused at any length.

**Readouts split what timing does not.** A prepared segment binds one receive
filter chain to each of its block positions, so a definition played with two
different ADCs becomes two segments — see
{ref}`shots that acquire nothing <acquire-nothing>` below for why that split
is made here and not in the period.

### Two decompositions

A spoiled gradient echo has nothing to reuse inside its own TR. Five blocks:
the four carrying waveforms are one segment, and the pad is the other, split
off because a pure delay is one position whose duration the playout sets. Two
segments, one instance each.

```{figure} ../assets/segments/gre_2d_tr_segments.png
One GRE TR, with the two segments shaded behind the waveforms.
```

MPRAGE repeats at the shot, not at the line, so its TR is the whole
inversion-recovery experiment — the inversion pulse and its crusher, the TI
fill, a train of spoiled gradient echoes, and the recovery delay that pads out
to the outer TR. Here that is 36 blocks over 135 ms, and three segments cover
them, played eleven times between them.

```{figure} ../assets/segments/mprage_3d_tr_segments.png
One MPRAGE TR: the inversion, the TI fill, eight instances of the readout
segment, and the recovery delay.
```

Segment 1 is played twice, as the 24 ms TI fill and as the 15 ms recovery pad:
two durations and one segment, because what the interpreter prepares is the
position and how long it waits there is a runtime parameter.

One instance of the readout segment is four blocks — excite, prewind, read,
rewind and spoil:

```{figure} ../assets/segments/mprage_3d_train_segments.png
One instance of the MPRAGE readout segment: excitation, prewinder, readout,
and the combined rewinder and spoiler.
```

Every instance occupies that same timing and almost nothing else about them
matches: the phase encode steps, the RF phase advances by the spoiling
increment, the rewinder undoes whatever the encode did. That difference is
exactly what the comparison excludes, which is why these are one segment and not
eight — and it is the form the scanner wants anyway.

(acquire-nothing)=
### Shots that acquire nothing

A preparation shot is blocks that carry no ADC, playing as the shot it stands in
for with the digitiser off. Two readouts of different length at one block
position are two ADC events playing as two prepared programs. The ADC is asked
about three separate times, and the answers are not the same.

```{figure} ../assets/segments/adc_identities.png
One block position, digitised differently or not at all, and the three
questions asked of it.
```

**What repeats leaves the ADC out.** Which readout a block digitises with is a
property of the instance, not of the rhythm the sequence repeats at. Counting
it would multiply the detected period by the number of readouts a position
holds — and every window-based safety check costs at least the square of the
period, so a two-echo readout would pay four times over for a distinction that
changes no waveform.

**What is prepared splits on it.** A prepared segment binds one receive filter
chain to each of its block positions, so two readouts that differ in sample
count or dwell cannot share one program. The partition is matched on *whether*
a position acquires, and the result is then split by the readouts each
repetition actually plays. Two echoes at one position give two segments, and
the scan alternates between them.

**A shot that acquires nothing joins either.** It digitises nothing there, so
neither segment's filter is used for it, and both play its gradients and RF
identically. The choice is unobservable, which is what makes it free — and it
is why a preparation shot never needs a segment of its own, and never
lengthens the repetition it precedes.

The middle answer has a cost: a position holding $k$ readouts is $k$ prepared
segments, and a scan that digitises differently on every repetition is refused
rather than turned into a segment table the size of the scan.

### Derivation instead of annotation

The PulSeg specification makes `TRID` annotation the segment-boundary
convention: the designer labels where segments begin. Pulserver does not read
it for that purpose, and this is a deliberate divergence rather than an
omission.

An annotation is a second source of truth. It can disagree with the content — a
designer edits the blocks and forgets the label — and then the interpreter is
obliged to believe something the sequence does not do. Every downstream
guarantee (the same block count per instance, the same structure, a segment that
can be prepared once) is a property of the *content*, so deriving the partition
from it makes those guarantees true by construction rather than by trust.

`TRID` still has a job, a narrower one: when a hyper-TR interleaves several
subsequences, it tells the RF checks which acquisitions belong to which
constituent TR, so SAR and coil heating are computed per constituent rather
than conservatively over the whole. It is never load-bearing — absent it, the
conservative check runs and the hardware monitor still stands.

Concatenated subsequences need no label at all: they are separate `.seq`
files linked by `NextSequence`, loaded as a collection and evaluated
independently.

## Two properties of the partition

**The period is anchored at the first block.** Periodicity is tested from the
start of the stream, so a scan whose opening differs from everything after it
— a catalyst ramp before a bSSFP train — comes back as one long TR. That is
what the file repeats: nothing shorter recurs from the first block onward, and
the catalyst is part of the unit. A single-slice train longer than 15 s
(`SINGLE_TR_MAX_DURATION_US`) is refused outright rather than segmented on a
period that was never verified.

**Granularity is bounded by the hardware, not chosen by taste.** Reading a scan
as one segment played once breaks no rule, and neither does the finest partition
the constraints allow. Pulserver takes the finest one that can actually be
played: no gradient can be broken off mid-ramp without violating the slew limit,
so the cuts fall only where the waveforms already permit playout to pause. The
result is the most reuse the scanner can get with every unit still physically
executable.

