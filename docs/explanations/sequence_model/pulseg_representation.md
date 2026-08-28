# The Pulserver PulSeg representation

{doc}`PulSeg <../background/pulseg>` describes what a scanner-side sequence
representation must carry: definitions separated from per-playout parameters,
blocks grouped into base blocks and reusable segments, and an ordered stream
of instances. Pulserver's scanner-side representation is a reading of that
specification, plus one structure the specification does not have. This page
says how the four structures map, what the addition is for, and where
Pulserver deliberately takes a different route; the companion page,
{doc}`tr_and_segmentation`, covers the added structure in full.

## The mapping

```{figure} ../assets/pulseg/pulserver_mapping.png
The four structures the specification defines, and the one Pulserver adds
above them. The addition is derived from the block content, which is what
keeps it from disagreeing with the sequence it describes.
```

Conversion happens in one pass: the interpreter parses the `.seq` and builds
the PulSeg structures directly, with no intermediate file. Every quantity the
specification's four structures carry is carried here:

| PulSeg structure | In Pulserver |
|---|---|
| `BaseBlock` — one block's definitions and duration | a row of indices into definition libraries: waveform shapes, trapezoid timings, ADC descriptions, RF envelopes — each stored once, however often it plays |
| `VirtualSegment` — an ordered list of base blocks | a segment found by {doc}`detection <tr_and_segmentation>`: a maximal run of blocks that repeats identically across TR instances |
| `SegmentInstance` — the per-playout parameters | instance tables: amplitudes, phase and frequency offsets, shot index, labels, and a reference into a shared rotation library |
| `ExecutionStream` — which instance plays when | runs (“positions *i*..*j* play instances *i*..*j* in order”) plus one period of the segment order |

The representation is lossless with respect to what plays: filling the
definitions back in and applying the instance parameters recovers the playout
the `.seq` describes, sample for sample.

## The addition: the structural TR

The specification's top level stops at the segment. It says which runs of
blocks are reusable units; it does not say what the scan is a repetition
*of*. Pulserver adds that one structure above the segment: the **structural
TR**, the smallest block period over which the normalized structure repeats —
six blocks for a gradient echo, the whole echo train for an FSE, a whole
shell for a ZTE. It is derived from the block content, and written back into the
`.seq` as `[DEFINITIONS] TRSize` so a consumer need not re-derive it.

It earns its place by paying for two things the four structures alone leave
expensive.

**Safety checks collapse onto one period.** SAR, gradient heating and the
acoustic drive are not properties of a block: each is a window sliding along
the whole scan, and evaluating one without knowing the period means sweeping
every block of a 30-minute protocol. Given the period, the sweep runs over a
single repetition and over the instance parameters that vary across its
occurrences — the worst B1rms rather than the first or the average, a
spectrum over a drive that is genuinely periodic. The {doc}`safety pages
<../safety/index>` are built on that reduction.

**Segmentation needs no annotation.** Segments are the TR partitioned by
structure: within one period, the maximal runs of blocks that repeat
identically across instances. Because the period follows from the content,
the boundaries the specification asks the designer to mark with `TRID` are
computed instead — the sequence labels its own repeating units, whether or
not its author labelled anything.

## Declared divergences from the specification

Pulserver conforms *semantically* — an interpreter reading its structures
sees exactly the objects the specification defines — but three choices are
declared rather than hidden:

- **Segmentation is derived from the content, not from `TRID`
  annotation.** The guarantees the specification asks the label to secure —
  every instance of a segment with the same block count and the same
  normalized structure — hold by construction when the partition is read off
  what the blocks play, rather than by trusting a label that a later edit can
  contradict. The full argument is in {doc}`tr_and_segmentation`.
- **The execution loop is stored compactly.** The specification's reference
  implementation writes one wide row per played block, with the 3×3 rotation
  matrix repeated inline. Pulserver keeps rotations in a shared library that
  instances reference, and stores the play order as runs — which is what
  makes a million-block scan fit in the tens of megabytes a scanner has to
  spare. Per-segment gradient energy likewise lives with the segment
  definition, where pulse generation reads it once, rather than on every
  instance row.
- **Times are integer microseconds** — the scanner's own clock — where the
  specification speaks seconds.

## What the definition/instance split buys

The split between definitions and instances is not bookkeeping; it is the
difference between a sequence that fits and one that does not. A definition
is *prepared* — converted to hardware units, resampled to the sequencer's
raster, loaded into pulse-generator memory — which is expensive and happens
once. An instance parameter is *applied* — written into a scan-loop row —
which is cheap and happens a million times. The
{doc}`performance pages <../performance/index>` measure what this buys; the
{doc}`safety pages <../safety/index>` lean on the same structure to make
whole-scan checks affordable.
