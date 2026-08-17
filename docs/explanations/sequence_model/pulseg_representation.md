# The Pulserver PulSeg representation

{doc}`PulSeg <../background/pulseg>` describes what a scanner-side sequence
representation must carry: definitions separated from per-playout parameters,
blocks grouped into reusable segments, and the repeating pattern — the TR —
made explicit. Pulserver's scanner-side representation is a reading of that
specification. This page says how the four structures map, and where
Pulserver deliberately takes a different route; the companion page,
{doc}`tr_and_segmentation`, covers the TR itself and how the segmentation is
found without anyone annotating it.

## The mapping

Conversion happens in one pass: the interpreter parses the `.seq` and builds
the PulSeg structures directly, with no intermediate file. Every quantity the
specification's four structures carry is carried here:

| PulSeg structure | In Pulserver |
|---|---|
| `BaseBlock` — one block's definitions and duration | a row of indices into definition libraries: waveform shapes, trapezoid timings, ADC descriptions, RF envelopes — each stored once, however often it plays |
| `VirtualSegment` — an ordered list of base blocks | a segment found by {doc}`detection <tr_and_segmentation>`: a maximal run of blocks that repeats identically across TR instances |
| `SegmentInstance` — the per-playout parameters | instance tables: amplitudes, phase and frequency offsets, shot index, labels, and a reference into a shared rotation library |
| the execution stream — which instance plays when | runs (“positions *i*..*j* play instances *i*..*j* in order”) plus one period of the segment order |

The representation is lossless with respect to what plays: hydrating the
definitions and applying the instance parameters recovers the playout the
`.seq` describes, sample for sample.

## Where Pulserver diverges, deliberately

Pulserver conforms *semantically* — an interpreter reading its structures
sees exactly the objects the specification defines — but three choices are
declared rather than hidden:

- **Segmentation is derived from the content, not from `TRID`
  annotation.** The specification asks the designer to label where segments
  begin; Pulserver finds the partition by comparing what the blocks actually
  play, so the same guarantees hold by construction rather than by trust, and
  unannotated files — every existing Pulseq sequence — work as they are. The
  full argument is in {doc}`tr_and_segmentation`.
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

The full field-by-field mapping, and the proposed `user_int[]`/`user_float[]`
amendment that would let an implementation carry its extras within the
specification, is in the conformance note shipped with the source.

## Conformance is tested, not asserted

The partition Pulserver derives must satisfy the same constraints the
specification places on a declared one — every instance of a segment with the
same block count and the same normalized structure. That is checked
mechanically: `tests/python/test_pulseg_oracle.py` validates the
interpreter's partition against an independent implementation of the
specification's rules, over the whole {doc}`sequence zoo
<../validation/sequence_zoo>`.

## Why a scanner wants this shape

The split between definitions and instances is not bookkeeping; it is the
difference between a sequence that fits and one that does not. A definition
is *prepared* — converted to hardware units, resampled to the sequencer's
raster, loaded into pulse-generator memory — which is expensive and happens
once. An instance parameter is *applied* — written into a scan-loop row —
which is cheap and happens a million times. The
{doc}`performance pages <../performance/index>` measure what this buys; the
{doc}`safety pages <../safety/index>` lean on the same structure to make
whole-scan checks affordable.
