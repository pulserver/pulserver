# Pulseq in one page

[Pulseq](https://pulseq.github.io) is an open file format for MR sequences.
A `.seq` file says what to play, on which axis, for how long — the complete
prescription of an experiment, portable across sites and vendors. Everything
Pulserver does sits on that description, so this is what it contains.

## Blocks and events

A sequence is an ordered list of **blocks**. A block has a duration and at
most one event per channel: one RF pulse, one gradient on each of `x`, `y`,
`z`, one ADC window, plus extensions. Events inside a block run concurrently,
each with its own delay from the block start; blocks run back to back with no
gap. There is no nesting, no loop construct and no branch — the file is the
flattened playout order.

```{figure} ../assets/pulseq/file_structure.png
A `.seq` file is one table of played blocks over four event libraries and a
shape library. Every cell of the block table is an id, and every id resolves
into a library entry that is written once however often it plays.
```

Events are stored in libraries and referenced by id, so an event used ten
thousand times is written once:

```
[BLOCKS]
# NUM  DUR  RF  GX  GY  GZ  ADC  EXT
    1  694   1   0   0   1    0    2
    2  356   0   2   3   4    0    0
    3  268   0   5   0   0    1    0
```

- **RF** carries an amplitude in Hz, plus ids into magnitude, phase and time
  shapes, a delay, frequency and phase offsets, and a `use` — excitation,
  refocusing, inversion, saturation, preparation.
- **Gradients** come two ways: a *trapezoid*, written as amplitude with rise,
  flat and fall times; or an *arbitrary* gradient, written as a shape id with
  an optional time shape when the samples are not on the raster.
- **ADC** is a sample count, a dwell time, a delay, and frequency and phase
  offsets.
- **Shapes** are compressed by a run-length encoding of their derivative,
  which is why a linear ramp of a thousand samples costs three numbers.

Times are on rasters the file declares in `[DEFINITIONS]`: an RF raster, a
gradient raster, an ADC raster, and a block-duration raster. Amplitudes are
in Hz and Hz/m — gyromagnetic-ratio-free, so the same file means the same
thing on any nucleus the scanner tunes to.

## Extensions

The extension chain is how a block carries anything the core format has no
column for. Each block points at a linked list of typed rows:

- **`LABELSET` / `LABELINC`** — counters and flags. Counters (`LIN`, `SLC`,
  `PAR`, `ECO`, `REP`, `AVG`, `SET`, `SEG`, `PHS`, `ACQ`) are the encoding
  indices a reconstruction sorts by; flags (`NAV`, `REV`, `IMA`, `NOISE`,
  `REF`, …) mark what an acquisition is for. Labels are *sticky*: a value set
  on one block holds until it is set again, so the file writes one row where
  a counter changes rather than one per block.
- **`ROTATIONS`** — a quaternion that rotates the block's gradients into the
  physical frame. One row per orientation instead of a rotated copy of every
  waveform, which is what makes a radial or spiral scan compact.
- **`TRIGGERS`** — wait on, or emit, a hardware signal.
- **`RF_SHIMS`** — per-transmit-channel magnitude and phase, for pTx.
- **`SOFT_DELAY`** — a delay whose duration the operator can adjust at the
  console without rewriting the file.

Pulserver adds no extension *types*; the two custom label names it does use
(`PMC`, `TRID`) ride the standard `LABELSET` row, and an interpreter that has
never heard of them ignores them, which is all a custom label can ever mean.

## Definitions, chains and the signature

`[DEFINITIONS]` is free-form key/value metadata. Some keys are conventional
(`FOV`, `Name`, the rasters, `TotalDuration`); the rest is whatever the
writer wants a reader to know. Pulserver reads a handful and writes a few of
its own — see {doc}`../sequence_model/tr_and_segmentation` for `TRSize`.

`NextSequence` links files into a chain: a calibration prescan and its
imaging scan are two files played in order, each with its own structure, not
one file with a mode flag. Pulserver loads a chain as a **collection** and
evaluates each subsequence independently.

`[SIGNATURE]` closes the file with an MD5 of everything above it, so a
scanner can refuse a file that changed after it was checked.

## What the format leaves open

Pulseq describes the playout completely. Between that description and a
sequence that stands in for a product one on a clinical scanner, three gaps
remain — none of them a flaw in the format, all of them the motivation for
the rest of this documentation.

```{figure} ../assets/pulseq/structure_levels.png
The same block list, read at the four levels a scanner-side representation
needs. The file states the order and nothing above it.
```

**Structure.** The file carries every event, but not how the events are
organised. At four levels, the information exists implicitly in the content
and no field carries it:

| Level | What the file does not say |
|---|---|
| within an event | which part is the fixed skeleton — a gradient's delay, its rise, flat and fall times, its time shape — and which varies playout to playout, the amplitude or the waveform shape it points at |
| the block | that the events playing together recur *as a unit* |
| the segment | that a reused ordered run of blocks is a unit a sequencer could prepare once |
| the TR | which periodic pattern of segments the scan is a repetition of |

Two consumers need that structure before they can do their job. A sequencer
that prepares a segment once and replays it — GE's, for one — has to
reconstruct the grouping before it can play the file at all. And the safety
quantities are not defined over a block: SAR, gradient heating and the
acoustic drive are each a window sliding along the whole scan, so evaluating
one without knowing the period means sweeping every block of it.

Where a repetition does exist, that sweep collapses onto a single period —
which is how the check becomes cheap enough for some vendors to run it that
way. A sequence is almost always periodic at some level, a hyper-TR if nothing
smaller, so recovering the pattern is where most of the saving comes from.

**Prescription-time adjustment.** A product sequence is edited at the
console: the operator changes TE, TR, FOV, matrix, orientation minutes before
the exam. Pulseq offers some of this — the `SOFT_DELAY` extension, FOV
scaling and repositioning of a written file — but the adjustable surface is
small compared to what an operator expects, and most parameter changes mean
rewriting the file.

**Design throughput.** The reference writers are MATLAB and Python. At
demonstration sizes that is comfortable; at clinical matrix sizes a scan is
hundreds of thousands to millions of blocks, and an interpreted loop over
them takes minutes where a console expects seconds.

Taken together, these three are what stands between a `.seq` file and a
replacement for a product sequence. The {doc}`sequence model
<../sequence_model/index>` pages describe how Pulserver closes each of them,
and {doc}`../safety/index` covers the checks that structure makes affordable.
