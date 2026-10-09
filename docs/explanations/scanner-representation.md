# PulSeg: what the scanner plays

```{admonition} TL;DR
:class: tldr

- Each Pulseq file becomes an IR cache in PulSeg's terms: base blocks, virtual segments prepared once, and a stream of segment instances.
- The repetition is pypulseqpp's; segments are cut within it where the gradients are near zero.
- The interpreter reads the cache through a C89 library pulserver ships.
```

A scanner plays faster and loads less when it prepares each distinct piece of
the sequence once and then only updates amplitudes, phases and rotations.
pulserver finds those pieces in each file of a `NextSequence` chain, in the
terms of PulSeg 2.1-alpha ([HarmonizedMRI/pulseg](https://github.com/HarmonizedMRI/pulseg)).
Lesson 3 of the {doc}`Course <../examples/course>` converts `gre2d` and looks at
its segments.

## What pulserver does

:::{container} capabilities

- **Converts a text or binary Pulseq chain into an IR cache that plays every block as designed.**

  Code: {func}`~pulserver.ir.convert`. Tests: *a converted cache reads back as the sequence it came from*; *the scanner plays every block as its file designs it*; *a binary file segments into the scan its text does*; *a chain plays its prescan file then its scan* (`test_ir.py`).
- **Takes the repetition pypulseqpp reports, and segments within it.**

  Code: `src/cpp/ir/structure.cpp`. Tests: *the ir segments the repetition pypulseqpp finds*; *a hyper tr declared as pypulseqpp writes it is the repeating unit* (`test_ir_repetition.py`).
- **Treats a sequence that does not repeat as one repetition, and refuses one longer than 15 s.**

  Code: `SINGLE_TR_MAX_DURATION_US`. Tests: *a sequence that does not repeat is one repetition*; *a sequence that does not repeat is refused past 15 s* (`test_ir_repetition.py`).
- **Cuts virtual segments only where every gradient is near zero, under a tunable grouping.**

  Code: {class}`~pulserver.ir.Grouping`. Tests: *the default grouping is what a conversion does unasked*; *a grouping moves the boundaries and not the gradient* (`test_ir_grouping.py`); *a gradient its file ends live may not end the repetition* (`test_ir.py`).
- **Gives repetitions that differ in pulses, timing or arbitrary gradients their own segment variant.**

  Code: `src/cpp/ir/`. Tests: *repetitions that play different pulses each play their own*; *repetitions whose gradients differ in timing each play their own* (`test_ir_segments.py`).
- **Streams waves through waveform memory, and refuses a conversion whose waves cannot load in time.**

  Code: {class}`~pulserver.ir.WaveBudget`. Tests: *waves that do not fit at once stream through a ring of slots per position*; *an instance that cannot load while the one before it plays is refused* (`test_ir_waves.py`).
- **Sets triggers, digital outputs and the `NOROT` exemption per block, as the labels state.**

  Code: `src/cpp/ir/`. Tests: *the scan waits for a cardiac trigger without driving an output*; *the scan pulses each output and leaves norot blocks unrotated* (`test_ir_playout.py`).
- **Moves the design to the field-of-view offset with RF and ADC frequency and phase offsets alone.**

  Code: {func}`~pulserver.ir.prescribe`. Tests: *a prescribed offset moves every readout of a file by one frequency*; *a prescribed slice offset moves every excitation by one frequency* (`test_ir.py`).
- **Writes the sound pressure levels and SAR ratios of each subsequence into the cache.**

  Code: {func}`~pulserver.ir.convert`. Tests: *each file of a chain carries its sound pressure levels into the cache*; *each file of a chain carries its sar ratios into the cache* (`test_ir.py`).
- **Ships the C89 reader and playout the interpreter links, tested 32-bit and vendor-tagged.**

  Code: `src/c/`, {func}`~pulserver.ir.play`. Tests: *a vendor cache written here loads in the scanner reader*; *a reader built for another vendor refuses the cache* (`test_ir.py`).

:::

## What the interpreter does

- **Prepares each virtual segment once and plays its instances**
  waves into waveform memory on the schedule the cache lays out.
- **Generates the scanner's instructions**
  waveforms onto the scanner's rasters ({doc}`safety-checks`).
- **Applies the prescription's rotation**

## How it works

### The PulSeg model

```{figure} ../_static/pulseg.svg
:figclass: only-light

Base blocks, virtual segments and the execution stream of segment instances.
```

```{figure} ../_static/pulseg-dark.svg
:figclass: only-dark

Base blocks, virtual segments and the execution stream of segment instances.
```

- A **base block** fixes the shapes and timing of its events at normalised amplitude; equal ones are stored once.
- A **virtual segment** is an ordered list of base blocks, such as an excitation and readout.
- A **segment instance** is one occurrence of a segment: gradient scales and rotation, swapped gradient waveforms, RF amplitude, offsets and shim, ADC offsets or no acquisition, triggers, outputs and block durations.
- The **execution stream** is the ordered list of instances, and it is the whole scan.

An instance plays $\mathbf{g}(t) = R\, s\, \hat{\mathbf{g}}(t)$: the base
block's normalised waveform, scaled by $s$ and rotated by $R$.

### Repetition and virtual segments

The repetition is the period `pypulseqpp.Sequence.repetition` reports.
Segments are cut within it where every gradient is within
`boundary_gradient_hz_per_m` of zero (100 Hz/m by default) and where `NOROT`
or `PMC` changes; navigators, edge delays and runs with other RF or ADC events
become segments of their own, up to 64 variants each
({class}`~pulserver.ir.Grouping`). The stream's segment IDs are run-length
encoded once per period.

```{figure} ../_static/repetition.svg
:figclass: only-light

One repetition of three segments played N times: IDs stored once, parameters per instance.
```

```{figure} ../_static/repetition-dark.svg
:figclass: only-dark

One repetition of three segments played N times: IDs stored once, parameters per instance.
```

### Non-periodic sequences and the field-of-view offset

A subsequence that does not repeat is one repetition, segmented the same way;
one longer than 15 s is refused, a limit of pulserver's playout memory model.
The offset is applied by {func}`~pulserver.ir.prescribe` as RF and ADC
frequency and phase offsets at the middle of each window; the remaining phase
under a varying readout gradient is applied by the proxy ({doc}`raw-data`).
Two offsets give two caches, and two designs.

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
