# PulSeg: what the scanner plays

```{admonition} TL;DR
:class: tldr

- The scanner does not read a Pulseq file while it plays. pulserver converts
  the file into an IR cache in PulSeg's terms: base blocks, virtual segments
  the scanner prepares once, and an execution stream of segment instances
  that set everything that changes from one TR to the next.
- The repetition is the one pypulseqpp reports. pulserver cuts it into
  virtual segments where the gradients are near zero, and stores the stream's
  segment IDs once per period.
- The cache is read on the scanner by a C89 library that pulserver ships and
  the interpreter links.
```

A Pulseq file lists every block of the scan. A scanner plays faster and loads
less when it prepares each distinct piece of the sequence once and then only
updates amplitudes, phases and rotations. pulserver finds those pieces for you:
it converts each file of a `NextSequence` chain into the IR cache the
interpreter loads, written beside the sequence file. The terms are those of
PulSeg 2.1-alpha ([HarmonizedMRI/pulseg](https://github.com/HarmonizedMRI/pulseg)).
Segment boundaries, the meaning of `TRID` and the use of periodicity are
pulserver's own.

Lesson 3 of the {doc}`Course <../examples/course>` converts `gre2d` and looks
at its segments and at what each TR changes.

## What pulserver does

:::{container} capabilities

- **Converts a Pulseq chain into an IR cache that plays every block as its file designs it.** Text and binary Pulseq files segment alike, and each file of the chain is a subsequence played in order.

  Code: {func}`~pulserver.ir.convert`. Tests: *a converted cache reads back as the sequence it came from*; *the scanner plays every block as its file designs it*; *a binary file segments into the scan its text does*; *a chain plays its prescan file then its scan* (`test_ir.py`).
- **Takes the repetition pypulseqpp reports, and segments within it.** A hyper-TR declared as pypulseqpp writes it is the repeating unit.

  Code: `src/cpp/ir/structure.cpp`. Tests: *the ir segments the repetition pypulseqpp finds*; *a hyper tr declared as pypulseqpp writes it is the repeating unit* (`test_ir_repetition.py`).
- **Treats a sequence that does not repeat as one repetition, and refuses one longer than 15 s.**

  Code: `SINGLE_TR_MAX_DURATION_US`. Tests: *a sequence that does not repeat is one repetition*; *a sequence that does not repeat is refused past 15 s* (`test_ir_repetition.py`).
- **Cuts virtual segments only where every gradient is near zero, under a grouping a vendor can tune.** Changing the grouping moves the boundaries, not the gradients.

  Code: {class}`~pulserver.ir.Grouping`. Tests: *the default grouping is what a conversion does unasked*; *a grouping moves the boundaries and not the gradient* (`test_ir_grouping.py`); *a gradient its file ends live may not end the repetition* (`test_ir.py`).
- **Gives a repetition that plays different pulses, timings or arbitrary gradients its own variant of a segment.**

  Code: `src/cpp/ir/`. Tests: *repetitions that play different pulses each play their own*; *repetitions whose gradients differ in timing each play their own* (`test_ir_segments.py`).
- **Plays rotated blocks as waves from waveform memory, streamed through a ring of slots when they do not fit at once.** A conversion whose waves cannot be loaded in time is refused before anything plays.

  Code: {class}`~pulserver.ir.WaveBudget`. Tests: *waves that do not fit at once stream through a ring of slots per position*; *an instance that cannot load while the one before it plays is refused* (`test_ir_waves.py`).
- **Sets triggers, digital outputs and the `NOROT` exemption per block, as the labels state.**

  Code: `src/cpp/ir/`. Tests: *the scan waits for a cardiac trigger without driving an output*; *the scan pulses each output and leaves norot blocks unrotated* (`test_ir_playout.py`).
- **Moves the design to the prescribed field-of-view offset with RF and ADC frequency and phase offsets, without changing a gradient.** Two offsets give two caches.

  Code: {func}`~pulserver.ir.prescribe`. Tests: *a prescribed offset moves every readout of a file by one frequency*; *a prescribed slice offset moves every excitation by one frequency* (`test_ir.py`).
- **Writes the sound pressure levels and SAR ratios of each subsequence into the cache.**

  Code: {func}`~pulserver.ir.convert`. Tests: *each file of a chain carries its sound pressure levels into the cache*; *each file of a chain carries its sar ratios into the cache* (`test_ir.py`).
- **Ships the C89 reader and playout the interpreter links, built 32-bit and vendor-tagged as on a scanner.** A reader built for another vendor refuses the cache.

  Code: `src/c/`, {func}`~pulserver.ir.play`. Tests: *a vendor cache written here loads in the scanner reader*; *a reader built for another vendor refuses the cache* (`test_ir.py`).

:::

## What the interpreter does

- **Prepares each virtual segment once and plays its instances**, loading
  waves into waveform memory on the schedule the cache lays out.
- **Generates the scanner's instructions**, including the resampling of
  waveforms onto the scanner's rasters ({doc}`safety-checks`).
- **Applies the prescription's rotation**, composed after each block's own.

## How it works

### The PulSeg model

```{figure} ../_static/pulseg.svg
:figclass: only-light

Base blocks with normalised waveforms; virtual segments as ordered lists of
base-block IDs; the execution stream as segment instances, each holding the
parameters of one occurrence.
```

```{figure} ../_static/pulseg-dark.svg
:figclass: only-dark

Base blocks with normalised waveforms; virtual segments as ordered lists of
base-block IDs; the execution stream as segment instances, each holding the
parameters of one occurrence.
```

The model is that of the PulSeg specification, version 2.1-alpha, sections 2
and 3 ([HarmonizedMRI/pulseg](https://github.com/HarmonizedMRI/pulseg)).

- A **base block** is a Pulseq block whose waveform amplitudes are normalised:
  it fixes the shapes and timing of its events but not their amplitudes.
  Equal base blocks are stored once.
- A **virtual segment** is an ordered list of base-block IDs, a reusable
  structural unit such as an excitation and readout or a preparation module.
  It need not be periodic.
- A **segment instance** is one occurrence of a virtual segment in the scan.
  It holds the concrete execution parameters: the amplitude scale of each
  gradient and its rotation, a gradient waveform swapped for another of the
  same timing, the RF amplitude, phase and frequency offsets and channel shim,
  the ADC phase and frequency offsets or no acquisition at all, the trigger
  waits and digital outputs, and the block durations.
- The **execution stream** is the ordered list of segment instances, and it
  defines the scan.

The physical gradient an instance plays is the base block's normalised
waveform $\hat{g}(t)$ scaled and rotated,

$$
\mathbf{g}(t) = R\, s\, \hat{\mathbf{g}}(t),
$$

with $s$ the instance's amplitude scale and $R$ its rotation. A scanner
prepares each virtual segment once, with its waveforms in waveform memory, and
plays every instance of it by updating these parameters. Every block of the
sequence is kept, in order, so the stream reproduces the sequence it was
converted from.

### Repetition and virtual segments

The cost of preparing the scan, and the size of the tables an interpreter
loads, grow with the number of distinct virtual segments and with how
irregular the stream is. Most sequences play one block pattern many times with
new amplitudes and phases, and conversion exploits that.

The repetition of a subsequence is the period `pypulseqpp.Sequence.repetition`
reports, starting at the first block; conversion does not search for one of its
own. The virtual segments are cut from that repetition at the block boundaries
the {class}`~pulserver.ir.Grouping` admits:

- a boundary may fall only where every gradient is within
  `boundary_gradient_hz_per_m` of zero, 100 Hz/m by default;
- a boundary must fall where the `NOROT` or `PMC` label changes, since the
  interpreter sets one prescription rotation per segment instance, and
  conversion fails when such a change falls under a gradient;
- navigator readouts, pure delays at the edge of a segment, and runs that play
  other RF or ADC events are split off as virtual segments of their own,
  according to the grouping, with at most 64 variants of a segment.

The virtual segments are then tiled over the execution stream. The sequence of
segment IDs is run-length encoded over one period, the shortest that the
whole stream is measured to repeat with; each instance keeps its own
parameters.

```{figure} ../_static/repetition.svg
:figclass: only-light

One repetition of three virtual segments, played N times: the segment IDs are
stored once per period, and each instance keeps its own parameters.
```

```{figure} ../_static/repetition-dark.svg
:figclass: only-dark

One repetition of three virtual segments, played N times: the segment IDs are
stored once per period, and each instance keeps its own parameters.
```

### Non-periodic sequences

A subsequence that does not repeat is one repetition, the whole subsequence.
It is segmented under the same grouping, its stream is still complete, and its
segment IDs are run-length encoded without a period. Conversion refuses a
single repetition longer than 15 s (`SINGLE_TR_MAX_DURATION_US` in
`src/cpp/ir/structure.cpp`). The limit belongs to pulserver's playout memory
model, not to PulSeg.

### Field-of-view offset

A pypulseqpp sequence is written in the logical frame, about the isocentre.
{func}`~pulserver.ir.prescribe` moves every file of the chain to the
prescribed field-of-view offset before conversion, without changing a
gradient: each RF pulse and ADC event receives the frequency and phase
offsets the offset requires, taken for a readout at the middle of its sampling
window. The cache keeps no ADC phase modulation, so where a readout gradient
varies across the window, the reconstruction proxy applies the remaining phase
from the trajectory, and from the header's `fov_offset_mm` when it states
another position ({doc}`reconstruction`). The prescription's rotation is not
applied to the cache: the scanner composes it after each block's own rotation.
Two offsets give two caches, and two designs ({doc}`architecture`).

### Differences from PulSeg 2.1-alpha

| Aspect | PulSeg 2.1-alpha | pulserver |
| --- | --- | --- |
| Segment-instance boundaries | Marked by the sequence designer with `TRID` labels (§4.2) | Found within the repetition under the {class}`~pulserver.ir.Grouping` |
| `TRID` | Identifies the virtual segment an instance plays | A sticky group, reported to the interpreter's checks with the duration of each occurrence; it places no segment boundary |
| Periodicity | Not part of the model | Taken from pypulseqpp when present, and used to encode the stream |
| Multishot gradient variants | One base block with shot variants, chosen per instance by the optional `gradient_shot_index` | No shot index: a gradient that differs between instances is played as a wave from waveform memory, or the run is a variant of its virtual segment |

## See it run

- {doc}`../generated/gallery/01-course/03_scanner_representation`: `gre2d`
  converted, its segments, and what changes from one TR to the next.
- {doc}`/generated/gallery/02-tours/02_segmentation`: the segmentation of the
  shipped sequences.
- {doc}`../developer-guide/internals/ir-cache`: the conversion passes, waves,
  the two stages of a playout and the cache file.
- {doc}`../api/ir`: the conversion interface.
