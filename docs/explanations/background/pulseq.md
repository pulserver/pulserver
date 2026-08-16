# Pulseq in one page

[Pulseq](https://pulseq.github.io) is an open file format for MR sequences.
A `.seq` file says what to play, on which axis, for how long — and nothing
about the scanner that will play it. Everything Pulserver does sits on that
description, so this is what it contains.

## Blocks and events

A sequence is an ordered list of **blocks**. A block has a duration and at
most one event per channel: one RF pulse, one gradient on each of `x`, `y`,
`z`, one ADC window, plus extensions. Events inside a block run concurrently,
each with its own delay from the block start; blocks run back to back with no
gap. There is no nesting, no loop construct and no branch — the file is the
flattened playout order.

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

## What the format does not say

Deliberately, and this is the point of it: nothing about hardware. No
gradient or slew limit, no coil, no SAR model, no acoustic response. A `.seq`
that plays on one system may be unplayable on another, and the file cannot
tell you which.

It also says nothing about *structure*: which blocks form the repeating unit,
which runs are reusable segments, where the echo is. That information exists
— it is implicit in the content — but no field carries it.

Those two gaps are the whole of Pulserver's job: {doc}`../safety/index` fills
the first against a system's actual limits,
{doc}`../sequence_model/tr_and_segmentation` fills the second by deriving the
structure from the blocks.
