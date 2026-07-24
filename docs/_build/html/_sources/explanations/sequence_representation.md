# How a sequence is represented

A `.seq` file is a flat list of blocks. A scanner needs something else: a
small set of reusable waveform definitions that fit in instruction memory,
plus a cheap description of how to replay them with per-instance amplitudes.
This page is about the representation in between, why it has the shape it
does, and what it costs you as a sequence author.

## The abstraction: PulSeg

[PulSeg](https://github.com/HarmonizedMRI/pulseg) is an abstract intermediate
representation for exactly this gap. Four concepts carry it:

| Concept | What it is |
| --- | --- |
| **BaseBlock** | A normalised Pulseq block: RF scaled to unit peak, gradients scaled per axis to unit peak, ADC copied as-is. Ids 0 and 1 are reserved for constant- and variable-duration pure delays. |
| **VirtualSegment** | An ordered list of base blocks that maps to one contiguous region of sequencer instruction memory, keyed by a `TRID` label. Repeated `TRID` means identical structure. |
| **SegmentInstance** | One playback of a segment, carrying the per-instance parameters: RF amplitude, phase and frequency; signed per-axis gradient scaling; rotation; ADC offsets; block durations. |
| **Execution stream** | The ordered traversal of segment instances that constitutes the scan. |

The important design decision is that the unit of the IR is the **segment**,
not the TR. A segment is a memory and reuse concept and assumes nothing about
periodicity. If the IR's fundamental unit *were* the TR it would be strictly
less expressive than Pulseq, which does not require sequences to be periodic
at all.

Pulserver realises all four concepts and adds one more — the **TR** — as an
optional overlay, discussed below. It never writes a PulSeg file: the `.seq`
is parsed straight into the vendor representation in one pass, and conformance
is a property of the object, demonstrable by projection. See
{doc}`../pulseg_conformance` for the clause-by-clause mapping.

## Deduplication: the first thing conversion does

Conversion starts by collapsing the block list. Every RF, gradient and ADC
event is normalised and hashed; identical definitions collapse to one entry,
and each raw block becomes a row of indices into those tables.

The compression is the entire economic case for the representation, and it is
large:

| Sequence | Raw blocks | Unique base blocks | Ratio |
| --- | ---: | ---: | ---: |
| GRE 64² × 1 slice | 512 | 8 | 64× |
| GRE 128² × 16 slices | 14 464 | 8 | 1808× |
| GRE 256² × 32 slices | 57 600 | 8 | 7200× |
| EPI 64² × 4 slices, ETL 16 | 337 | 9 | 37× |
| FSE 256² × 20 slices, ETL 16 | 22 080 | 9 | 2453× |
| bSSFP 128², 64 frames | 16 576 | 7 | 2368× |
| MPRAGE 192² × 128 partitions | 181 248 | 10 | 18 125× |

Note that the unique-block count is flat as the matrix grows: GRE goes from
512 blocks to 57 600 — 112× more sequence — at a constant 8 definitions.
Phase encoding does not create new definitions, because a scaled gradient is
the same definition at a different per-instance amplitude — every distinct
`ky`/`kz` step reuses one gradient shape, scaled — rather than each step
soliciting its own trapezoid solve. That is why the scanner-side memory in
the tables further down barely moves with matrix size, and why the count above
is single digits per sequence family rather than growing with resolution,
slice count or echo-train length.

**Multishot gradients.** When a gradient's *shape* varies per instance but its
*timing* does not, the two cases are distinguished:

- Shapes related by scaling or rotation — phase-encode tables, radial spokes,
  rotated spiral interleaves — collapse to a **single** definition, varied by
  signed amplitude and rotation.
- Genuinely independent co-timed shapes — independently optimised spiral
  interleaves — become **multiple shot variants** of one multishot definition
  sharing a timing structure, selected per instance by shot index.

Timing identity is `(delay, rise, flat, fall)` for trapezoids, `(delay,
time-shape-id)` for extended trapezoids, `(delay, num_samples)` for
uniform-raster arbitrary gradients.

## TR detection

The TR is not in the file. Pulseq has no TR concept, and Pulserver needs one:
worst-case RF and SAR limits are stated per TR, and the periodicity-exploiting
analyses ([mechanical resonance](mechanical_resonance_safety.md),
[PNS](pns_safety.md)) evaluate one canonical window rather than a whole scan.

Detection runs on the imaging region only — prep and cooldown are stripped
first — and looks for the **shortest period starting at offset 0**. Two passes:

1. **Timing/id pattern.** Each block position becomes a token: its duration
   if the duration is fixed, or the negated definition id if it is variable.
   The shortest $l$ with `s[i] == s[i+l]` for all $i < l$ wins, verified
   across the whole region.
2. **Structural fallback.** If no exact token period exists, the search runs
   again on block *structure* — same duration, and the same pattern of which
   event slots are occupied — which recovers the period when interleaves give
   different definition ids to structurally identical blocks.

The period search deliberately starts at offset 0 rather than scanning all
offsets. Searching from arbitrary offsets finds short sub-patterns that do not
span the region — `[rephaser, nav, rephaser, nav]` inside an EPI readout train
is a real example — and would report a TR shorter than any TR.

If neither pass finds a period, the whole imaging region is one TR. That is
the correct answer for a genuinely aperiodic sequence, and it is why the
analyses still work on one: they just get a large window.

Prep and cooldown are then compared against the discovered pattern.
"Degenerate" means structurally identical to the body — the sequence has no
distinct preparation — which is what lets the safety analyses use one imaging
TR as their window instead of a whole pass.

## Segment detection

A segment must be loadable as one contiguous unit of instruction memory, so
its boundaries are constrained by hardware, not by taste. Two rules:

**Gradients must be zero at a cut.** A boundary is only a candidate if the
last sample of the preceding block and the first sample of the following block
are both below 100 Hz/m on all three axes. The threshold is a fixed constant,
not derived from `max_slew`, so that segmentation is a property of the
sequence and not of the scanner it happens to be checked against — the same
`.seq` must segment identically everywhere.

**Cuts go where a new excitation begins.** A state machine walks the TR
looking for the last zero-gradient candidate *before* each RF, so a segment
starts at the excitation that owns it rather than in the middle of the
preceding readout:

- `SEEKING_FIRST_ADC` — before the first acquisition, remember the last
  candidate seen before each RF; on reaching the first ADC, cut there.
- `SEEKING_BOUNDARY` — on each subsequent RF, cut at the last candidate since
  the previous cut.
- `OPTIMIZED_MODE` — reached when an RF arrives with no candidate boundary
  since the last cut (the gradients never returned to zero). No further cuts;
  the rest of the TR is one segment.

That third state is the honest answer to a sequence that never gives the
segmenter a legal cut — a continuous bSSFP readout, say. It produces one large
segment rather than an illegal one.

Pure-delay segments are then stripped, and delays whose duration varies across
instances are marked adjustable so they can be applied per instance at scan
time rather than baked in.

## Segment deduplication across subsequences

A collection may hold several subsequences — a localiser then a scan, a
calibration then the acquisition — and they often share structure. After
per-subsequence segmentation, a global pass collapses segments whose
**materialised instruction memory is identical**, and builds a local → global
remap.

Equality here is deep and deliberately strict, because two segments that map
to one memory region must be interchangeable in every respect the hardware can
observe: same block count, same navigator flag, same trigger type, and for
every block the same duration, the same RF definition, the same per-axis
gradient definitions, and the same ADC.

The one exemption is instructive. A **dynamic** adjustable pure-delay block —
no RF, gradient or ADC, no digital output, no rotation, and a duration that
genuinely varies across its own subsequence's instances — is compared without
its duration, because that duration is applied per instance at scan time
anyway. A *static* adjustable delay is not exempt: nothing sets its period at
scan time, so its baked duration must still match exactly. Collapsing those
two cases together would produce a merged definition that silently plays the
wrong duration, and the exemption's conditions have to match the scan-time
"is this delay variable?" test exactly, or the two disagree.

## What may vary between TR instances

This is the practical question for a sequence author, and the answers are
enforced rather than advisory — `pulseg_check_consistency` runs inside
conversion, so a sequence that breaks a rule fails to load at all.

| Varies across TR instances | Accepted | Consequence |
| --- | --- | --- |
| Signed gradient amplitude scaling | yes | The phase-encode table. One definition, per-instance scale. |
| RF phase offset | yes | RF spoiling, phase cycling. |
| RF frequency offset | yes | Slice selection across a multi-slice loop. |
| Per-block rotation | yes | Radial spokes, rotated interleaves, oblique prescriptions. |
| Pure-delay duration | yes, with a caveat | TE/TR fill. A duration that changes *monotonically* across TRs breaks the timing period, so TR detection falls back to one large TR — legal, but the analyses lose their small window. |
| **RF amplitude (variable flip angle)** | **no** | Rejected: `RF amplitude pattern is not periodic across canonical TRs`. |
| **RF shim weights** | **no** | Rejected by the same rule, on shim id. |

The two rejections have the same cause and are worth stating plainly, because
they are the sharpest constraint the representation imposes.

Worst-case RF and SAR limits are computed **per canonical TR**. If the RF
amplitude pattern is not periodic, there is no canonical TR to compute them
over — the worst TR is not the one that was measured, and no amount of
per-instance metadata recovers that, because the limit is an integral over a
window whose contents changed. The check therefore compares every TR instance
against the reference TR, block by block, and rejects on the first mismatch.

The workable pattern today is to express a variable-flip-angle acquisition as
**separate subsequences**, one per distinct RF pattern. Each is internally
periodic, each gets its own worst-case analysis, and segment deduplication
collapses whatever structure they share back into one region of instruction
memory — so the cost of the split is bookkeeping, not memory.

Everything in the "yes" column, by contrast, leaves the RF energy of a TR
unchanged, which is why it is free.

## The cache, and why it has sections

Conversion is the expensive step. The result is written to a binary cache
beside the `.seq` (`.pseg` by default, vendor-selectable), so it is paid once
per prescription rather than once per scan.

But the consumers of that cache are not the same machine. Sections exist
because of who reads them:

| Section | Holds | Scales with |
| --- | --- | --- |
| `COMMON` | Collection and subsequence metadata, segment definitions, TR descriptor | Sequence structure |
| `SHAPES` | The deduplicated RF, gradient and ADC waveform library | Unique definitions |
| `INSTANCES` | Per-instance block, RF, gradient and ADC tables | **Scan length** |
| `ROTATIONS` | Per-instance rotation matrices | **Scan length** |
| `SCANLOOP` | The execution stream | **Scan length** |
| `DEFINITIONS` | The `.seq` `[DEFINITIONS]` key/value pairs | Nothing |
| `TRAJECTORY` | k-space trajectory and encoding spaces, for reconstruction | Scan length |
| `SEQDESC` | Event list and RF shape metadata, for reconstruction | Scan length |
| `FREQMOD` | The frequency-modulation plan | Scan length |
| `VENDOR` | An opaque vendor blob, written only if a callback supplies one | — |

Without disclosing any vendor's internals, the interpreter runs a sequence in
two distinct phases, and they need different halves of that table:

**Pulse generation** builds the hardware waveform images for each segment.
It needs the waveform library and the segment definitions — and nothing that
scales with scan length, because everything it resolves per (segment, block
position) is frozen into the segment definitions' initial-state records at
parse time. `pulseg_load_geninstructions_cache` therefore reads **COMMON +
SHAPES only**.

**The scan loop** walks the execution stream, setting per-instance amplitudes,
phases, frequencies and rotations as it goes. It needs all of it:
`pulseg_load_scanloop_cache` reads COMMON + INSTANCES + ROTATIONS + SHAPES +
SCANLOOP.

The split is not a micro-optimisation. Pulse generation runs where memory is
scarcest, and the sections it does not read are precisely the ones that grow
without bound as a scan gets longer. Measured on the C library with an
interposed allocator:

| Sequence | Cache file | Pulsegen load | Scan-loop load | Ratio |
| --- | ---: | ---: | ---: | ---: |
| GRE 128² × 16 slices | 4.7 MB | 30 KB | 4.5 MB | 152× |
| MPRAGE 192² × 128 partitions | 925 MB | 143 KB | 486 MB | 3 400× |

The pulsegen-stage figure is essentially flat — tens to hundreds of kilobytes
across the whole zoo, because it is bounded by the definition count, which
dedup already showed barely grows. Full numbers and timings are in
{doc}`performance`.

That is also the clearest way to see why the base-block dedup at the top of
this page matters: it is what makes the pulse-generation working set a
function of how many *distinct shapes* a sequence contains, rather than of how
long it runs.
