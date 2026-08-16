# PulSeg: reusable definitions plus instances

[PulSeg](https://github.com/HarmonizedMRI/pulseg) is an abstract intermediate
representation for the gap Pulseq leaves open: a small set of reusable
waveform definitions that fit in scanner instruction memory, plus a cheap
description of how to replay them with per-instance amplitudes. Four concepts
carry it:

| Concept | What it is |
| --- | --- |
| **BaseBlock** | A normalised Pulseq block: RF scaled to unit peak, gradients scaled per axis to unit peak, ADC copied as-is. Ids 0 and 1 are reserved for constant- and variable-duration pure delays. |
| **VirtualSegment** | An ordered list of base blocks that maps to one contiguous region of sequencer instruction memory, keyed by a `TRID` label. Repeated `TRID` means identical structure. |
| **SegmentInstance** | One playback of a segment, carrying the per-instance parameters: RF amplitude, phase and frequency; signed per-axis gradient scaling; rotation; ADC offsets; block durations. |
| **Execution stream** | The ordered traversal of segment instances that constitutes the scan. |

The unit of the IR is the **segment**, not the TR. A segment is a memory and
reuse concept and assumes nothing about periodicity — Pulseq does not require
sequences to be periodic, so an IR whose fundamental unit *were* the TR would
be strictly less expressive. Pulserver realises all four PulSeg concepts and
adds a TR overlay on top of them; see {doc}`pulserver` for why that overlay
exists and what it costs a sequence author.

Pulserver never writes a PulSeg file: the `.seq` is parsed straight into this
representation in one pass, and conformance is a property of the resulting
object, demonstrable by projection — see {doc}`../../pulseg_conformance` for
the clause-by-clause mapping.

## Deduplication: the first thing conversion does

Conversion starts by collapsing the block list. Every RF, gradient and ADC
event is normalised and hashed; identical definitions collapse to one entry,
and each raw block becomes a row of indices into those tables.

| Sequence | Raw blocks | Unique base blocks | Ratio |
| --- | ---: | ---: | ---: |
| GRE 64² × 1 slice | 512 | 8 | 64× |
| GRE 128² × 16 slices | 14 464 | 8 | 1808× |
| GRE 256² × 32 slices | 57 600 | 8 | 7200× |
| EPI 64² × 4 slices, ETL 16 | 337 | 9 | 37× |
| FSE 256² × 20 slices, ETL 16 | 22 080 | 9 | 2453× |
| bSSFP 128², 64 frames | 16 576 | 7 | 2368× |
| MPRAGE 192² × 128 partitions | 181 248 | 10 | 18 125× |

The unique-block count is flat as the matrix grows: GRE goes from 512 blocks
to 57 600 — 112× more sequence — at a constant 8 definitions. Phase encoding
does not create new definitions, because a scaled gradient is the same
definition at a different per-instance amplitude — every distinct `ky`/`kz`
step reuses one gradient shape, scaled — rather than each step soliciting its
own trapezoid solve. This is why scanner-side memory barely moves with matrix
size (see {doc}`../benchmarks`), and why the unique-definition count above is
single digits per sequence family rather than growing with resolution, slice
count or echo-train length.

**Multishot gradients.** When a gradient's *shape* varies per instance but
its *timing* does not, two cases are distinguished: shapes related by scaling
or rotation — phase-encode tables, radial spokes, rotated spiral interleaves
— collapse to a **single** definition, varied by signed amplitude and
rotation; genuinely independent co-timed shapes — independently optimised
spiral interleaves — become **multiple shot variants** of one multishot
definition sharing a timing structure, selected per instance by shot index.
Timing identity is `(delay, rise, flat, fall)` for trapezoids, `(delay,
time-shape-id)` for extended trapezoids, `(delay, num_samples)` for
uniform-raster arbitrary gradients.

## Segments are inferred, not authored

A segment must be loadable as one contiguous unit of instruction memory, so
its boundaries are constrained by hardware, not by taste
(`csrc/src/structure/pulseg_structure.c`). Two rules:

**Gradients must be zero at a cut.** A boundary is only a candidate if the
last sample of the preceding block and the first sample of the following
block are both below a fixed 100 Hz/m on all three axes
(`SEG_ZERO_GRAD_THRESHOLD_HZ_PER_M`). The threshold is a constant, not
derived from `max_slew`, so segmentation is a property of the `.seq` file and
not of the scanner it happens to be checked against.

**Cuts go where a new excitation begins.** A state machine walks each
subsequence looking for the last zero-gradient candidate *before* each RF, so
a segment starts at the excitation that owns it rather than in the middle of
the preceding readout:

- `SEEKING_FIRST_ADC` — before the first acquisition, remember the last
  candidate seen before each RF; on reaching the first ADC, cut there.
- `SEEKING_BOUNDARY` — on each subsequent RF, cut at the last candidate since
  the previous cut.
- `OPTIMIZED_MODE` — reached when an RF arrives with no candidate boundary
  since the last cut (the gradients never returned to zero). No further
  cuts; the rest of the region is one segment.

That third state is the honest answer to a sequence that never gives the
segmenter a legal cut. Pure-delay segments are then stripped, and delays
whose duration varies across instances are marked adjustable so they can be
applied per instance at scan time rather than baked in.

`Sequence.segments` exposes the partition used by the scanner; each entry
resolves to its maximum-energy instance, so plotting it answers the useful
question: which waveform does this reusable instruction region actually have
to carry? The figures below are exactly that query against the C library —
not illustrative redraws — run over the shipped example zoo.

**GRE and plain EPI: one segment.** A GRE TR is one excitation followed by
one readout with no intervening RF, so there is only ever one segment to
find. EPI's blip train keeps the phase-encode gradient non-zero between
successive lines, so `SEGSTATE_OPTIMIZED_MODE` never gets a second legal cut
either — the whole excitation-plus-train is one segment:

![GRE inferred segment](../assets/segments/gre_2d_segment_0.png)

![EPI inferred segment](../assets/segments/epi_2d_segment_0.png)

**FSE: also one segment, for the same structural reason.** A CPMG echo
train's crusher pairs around each refocusing pulse are deliberately kept
non-zero across the transition — collapsing them to exact zero would risk a
stimulated-echo pathway — so the segmenter never sees a legal cut between the
excitation and the first refocusing pulse, or between refocusing pulses. One
69-block segment is the structurally correct answer for this crusher design,
not a segmenter gap:

![FSE inferred segment](../assets/segments/fse_2d_segment_0.png)

**MPRAGE: two segments.** An inversion prep is structurally unlike the SPGR
body that follows it, and the spoiler after the adiabatic pulse *does* return
every axis to zero before the readout loop starts, so the segmenter finds a
real cut. `Sequence.segments` reports segment 0 (2 blocks, 42.0 ms — the
inversion pulse and its spoiler) and segment 2 (4 blocks, 13.4 ms — one SPGR
readout TR, structurally identical to a plain GRE segment) as the two real
segments, with a pure-delay segment between them for the inversion-recovery
wait:

![MPRAGE inversion segment](../assets/segments/mprage_2d_segment_0.png)

![MPRAGE readout segment](../assets/segments/mprage_2d_segment_2.png)

**EPI with a fat-saturation module: two segments.** The fat-sat pulse's own
spoiler is a legal cut in exactly the same way MPRAGE's inversion spoiler is,
so enabling `fat_sat=True` splits the fat-sat module (segment 1: 2 blocks,
8.62 ms) from the excitation-plus-readout train (segment 2) that used to be
the whole sequence on its own:

![EPI+fat-sat: fat-sat segment](../assets/segments/epi_2d_fatsat_segment_1.png)

![EPI+fat-sat: readout segment](../assets/segments/epi_2d_fatsat_segment_2.png)

Reproduce these with `python docs/_bench/segment_plots.py`.

## Segment deduplication across subsequences

A collection may hold several subsequences — a localiser then a scan, a
calibration then the acquisition — and they often share structure. After
per-subsequence segmentation, a global pass collapses segments whose
**materialised instruction memory is identical**, and builds a local → global
remap.

Equality here is deep and deliberately strict, because two segments that map
to one memory region must be interchangeable in every respect the hardware
can observe: same block count, same navigator flag, same trigger type, and
for every block the same duration, the same RF definition, the same per-axis
gradient definitions, and the same ADC.

The one exemption is instructive. A **dynamic** adjustable pure-delay block —
no RF, gradient or ADC, no digital output, no rotation, and a duration that
genuinely varies across its own subsequence's instances — is compared without
its duration, because that duration is applied per instance at scan time
anyway. A *static* adjustable delay is not exempt: nothing sets its period at
scan time, so its baked duration must still match exactly. Collapsing those
two cases together would produce a merged definition that silently plays the
wrong duration.
