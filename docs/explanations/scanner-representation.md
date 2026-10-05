# The scanner IR

A Pulseq file lists every block of a scan in play order, with each event stored
once in a library and referenced by id. A scanner's pulse generator is
programmed differently: waveform memory is allocated once per distinct
waveform, and a scan is played as repetitions of a small number of instruction
sequences, with amplitudes, phases and rotations updated between repetitions.
Pulserver computes this intermediate representation (IR) of a sequence on the
host and writes it into a binary cache beside the sequence file. The
interpreter loads the cache rather than parsing the Pulseq file.

## Subsequences

{func}`~pulserver.ir.convert` reads the `NextSequence` chain starting at a
sequence file, text or binary, with `pypulseqpp.Sequence`. Each file of the
chain is a subsequence of one scan, played in order; a prescan and the imaging
sequence are typically two subsequences. Each subsequence is analysed on its
own and the results are chained into one collection.

## Prescription

A pypulseqpp sequence is written in the logical frame, about the
isocentre. The prescribed field-of-view offset $\mathbf{d}$ reaches the host in
the `fov_offset_x`, `fov_offset_y` and `fov_offset_z` entries of the protocol,
in mm along the logical readout, phase-encoding and slice axes, and
{func}`~pulserver.ir.prescribe` applies it to every file of the chain before
the passes run. No gradient is changed:

- an RF pulse played under a gradient $G$ along an axis of the offset receives
  the frequency offset $G d$, with $G$ in Hz/m, and a phase offset; where the
  gradient varies during the pulse, the remaining phase is added as a phase
  shape;
- a readout receives the frequency and phase offsets, and where the gradient
  varies during it a phase modulation, so that the phase of each sample
  relative to its excitation is $2\pi\,\mathbf{d}\cdot\mathbf{k}(t)$, with
  $\mathbf{k}$ the k-space location of the sample in 1/m;
- blocks labelled `NOPOS` keep the phases they were designed with.

A block that carries a rotation extension plays its drawn gradients turned by
that rotation, and it is moved by the gradients it plays.

The cache carries each readout's frequency and phase offsets, taken at the
middle of its sampling window, and no phase modulation: the scanner
demodulates by the offsets alone. The reconstruction proxy applies the rest,
$2\pi\,\mathbf{d}\cdot(\mathbf{k}(t) - \mathbf{k}(t_c) - \dot{\mathbf{k}}(t_c)(t - t_c))$
about the window centre $t_c$, from the readout's k-space and the offset
$\mathbf{d}$ the design was converted at. Where the MRD header states the
object at another position (`fov_offset_mm`), such as one a motion correction
updates, the proxy also applies $2\pi\,\Delta\mathbf{d}\cdot\mathbf{k}(t)$
for the difference, which needs no new cache. The reconstruction then receives an object at
the stated position at the centre of its field of view. A chain whose stored
modulation does not hold one phase per ADC sample is refused. The rotation of the prescription is not applied
to the cache: the scanner plays it through its rotation matrix, composed after
each block's own rotation. Two offsets make two caches of one design, and two
designs ({doc}`designs`).

## Passes

| Pass | Result |
| --- | --- |
| Event deduplication | A library of distinct RF, gradient and ADC definitions, and a per-block instance table recording the definition each block plays and its amplitude |
| Repetition | The repeating unit of each subsequence and its repetition time (TR): the unit pypulseqpp's `Sequence.repetition` finds, from the first block; a subsequence that does not repeat is one repetition, refused when it is longer than 15 s |
| Segmentation | The repeating unit divided into segments at block boundaries where every gradient waveform is zero |
| Execution stream | The order in which segments are played over the whole scan |
| Label table | The Pulseq labels in force at every readout, three of which fill the ADC label columns |
| RF statistics | For each RF definition: its transmit channels, flip angle and energy, as pypulseqpp counts them; its duration from the envelope; and the bandwidth and bands of a multiband pulse from its spectrum |
| Gradient statistics | For each gradient definition: the range of amplitudes its instances play; and for each shape it plays, the steepest slew rate and the integrals of the squared waveform and of its squared slew rate, pypulseqpp's `Sequence.gradient_statistics` over the amplitude that plays the shape |

The limits and rasters of the scanner (`pypulseqpp.Opts`) are those under which
the scan is segmented. A segment boundary falls where every gradient is zero,
judged by the first and last values each arbitrary gradient's library row
stores for the event's edges; a trapezoid starts and ends at zero. A segment
is prepared from the RF and gradient definitions of the blocks of one
repetition; a block instance sets only their amplitudes, frequency and phase
offsets and gradient shape, or, where the blocks carry a rotation, the rotated
wave it plays. Every repetition that plays a segment
therefore plays the same RF and gradient definitions at each position, and the
same ADC definition wherever it acquires. A repetition that plays other
definitions, such as a pulse of its own per shot in a repetition pypulseqpp
finds by block duration and the channels played, or that digitises with other
ADC events, plays a segment of its own; a segment with more than 64 such
variants of either kind is refused. The
spectral statistics are measured by pypulseqpp's
`calc_rf_bandwidth` when the chain is read, with the Pulseq recipe of the width
at half the spectral peak; a band is a run of the spectrum above 30 % of its
peak, and its offset is measured from the carrier. A dynamic pTx pulse holds
its channels one after another over one time base. Its channel count is
pypulseqpp's `Sequence.rf_channels`: the number of samples at its first sample
time, when the times are that many identical copies. Its flip angle is
pypulseqpp's `Sequence.rf_flip_angles`, the channels' integrals summed
coherently, the flip where every channel has unit, in-phase sensitivity; the
envelope for the power statistics is the root sum of squares of the channels.
The energy is held as the integral of the squared envelope scaled to unit
peak, in seconds: pypulseqpp's `calc_rf_power` energy, the integral of
$|b_1|^2$, over its peak power, both summed over the channels.
A pulse a file leaves unlabelled takes the use pypulseqpp detects for it when
the chain is read. RF and ADC frequency and phase offsets are stored absolute:
the ppm offsets are resolved on the host, at the gamma and B0 of the call's
limits, by pypulseqpp's `SequenceLibraries.absolute_offsets`. The labels
and flags in force at each block, with `PMC` in force from the first, are
pypulseqpp's `Sequence.evaluate_labels`, and a `TRID` group starts at each block
`Sequence.label_blocks` lists as setting it; the rotation and RF shim a block
plays are `Sequence.block_rotations` and `Sequence.block_shims`. A segment
position records whether the scanner may move its excitation at run time by a
carrier offset alone: every instance plays its RF pulse under one gradient,
steady from the pulse's first sample to its last as
`Sequence.rf_gradients` finds it, in a block without a rotation.
`label_column_map` selects the three labels the
interpreter records per readout, as indices in the order SLC, PHS, REP, AVG,
SEG, SET, ECO, PAR, LIN, ACQ.

## See also

* {doc}`../api/ir` — the conversion interface.
* {doc}`protocol` — the protocol entries the offset arrives in.
* [`src/c/include/pulseg/`](https://github.com/pulserver/pulserver/tree/main/src/c/include/pulseg)
  — the public C headers the interpreter includes.
* {doc}`/generated/gallery/02-tours/02_segmentation` — the segmentation of shipped sequences, executed.
