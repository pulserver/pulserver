# IR cache

The implementation of the scanner representation
({doc}`../../explanations/scanner-representation`): how the prescription is
applied, the passes that compute it and the statistics they record, how gradients that a
prepared segment cannot play are carried as waves, how a playout loads the
cache in two stages, the gradients kept for the scanner's heating model, and
the layout of the cache file. The C headers an interpreter
includes are in
[`src/c/include/pulseg/`](https://github.com/pulserver/pulserver/tree/main/src/c/include/pulseg).

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
designs ({doc}`../../explanations/designs`).

## Conversion passes

| Pass | Result |
| --- | --- |
| Event deduplication | A library of distinct RF, gradient and ADC definitions, and a per-block instance table recording the definition each block plays and its amplitude |
| Repetition | The repetition of each subsequence and its repetition time (TR): the period pypulseqpp's `Sequence.repetition` finds, from the first block; a subsequence that does not repeat is one repetition, refused when it is longer than 15 s |
| Segmentation | The repetition divided into virtual segments at block boundaries the {class}`~pulserver.ir.Grouping` admits |
| Execution stream | The segment instances of the whole scan, in play order |
| Label table | The Pulseq labels in force at every readout, three of which fill the ADC label columns |
| RF statistics | For each RF definition: its transmit channels, flip angle and energy, as pypulseqpp counts them; its duration from the envelope; and the bandwidth and bands of a multiband pulse from its spectrum |
| Gradient statistics | For each gradient definition: the range of amplitudes its instances play; and for each shape it plays, the steepest slew rate and the integrals of the squared waveform and of its squared slew rate, pypulseqpp's `Sequence.gradient_statistics` over the amplitude that plays the shape |

The limits and rasters of the scanner (`pypulseqpp.Opts`) are those under which
the scan is segmented. A segment boundary falls where every gradient is
within the grouping's tolerance of zero, judged by the first and last values
each arbitrary gradient's library row stores for the event's edges; a
trapezoid starts and ends at zero. A virtual segment is prepared from the RF and gradient definitions of the blocks of one
repetition; a block instance sets only their amplitudes, frequency and phase
offsets and gradient shape, or, where the blocks carry a rotation, the rotated
wave it plays. Every repetition that plays a virtual segment
therefore plays the same RF and gradient definitions at each position, and the
same ADC definition wherever it acquires. A repetition that plays other
definitions, such as a pulse of its own per shot in a repetition pypulseqpp
finds by block duration and the channels played, or that digitises with other
ADC events, plays a virtual segment of its own; one with more than 64 such
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

## Waves

A playout prepares the gradient events of each segment position once, before
the scan, and each segment instance sets only their amplitudes. Two kinds of
position cannot be played that way. A rotation extension turns a block's
gradients within the logical frame, so the waveform each axis plays combines
the block's three gradient events. And where the instances of a position play
a gradient definition or shape other than the one its events are prepared
with, as the interleaves of a spiral drawn as distinct shapes do, no amplitude
makes the prepared event play them. At both, a block plays one waveform per
axis from waveform memory: a wave. Logical axis $o$ plays

$$
g_o(t) = \sum_d R_{od}\, a_d\, w_d(t) = m \sum_d R_{od}\, \frac{a_d}{m}\, w_d(t),
$$

with $R$ the block's rotation, the identity for a block without one, $a_d$ the
amplitude of the event on axis $d$, $w_d$ its waveform normalised to a largest
magnitude of one, and $m$ the $a_d$ of largest magnitude, sign included. The
sum on the right is fixed by the events' definitions and shapes, the rotation
and the amplitude ratios $a_d/m$; an instance only scales it by $m$. The
conversion keeps each distinct sum once per subsequence, with ratios that
round to the same multiple of $10^{-4}$ taken as equal, normalised on each
axis to a largest magnitude of one. Blocks that differ in amplitude or
polarity alone share one wave.

At every segment position where an instance carries a rotation other than
the identity, or plays a gradient the position's events are not prepared
with, each block with a gradient event plays its wave on each axis, at $m$
times the wave's largest magnitude there, and the scanner applies the
prescription's rotation alone, or no rotation under `NOROT`. Each such
position records the number of points of the longest wave it plays, and the
span of time the waves cover. Every wave a position plays, and every position
a wave plays at, is given the union of those spans, so that the waves one
position plays all cover one interval and a playout can prepare one waveform
of that length for it.

A wave that combines trapezoids and arbitrary gradients with time shapes is
piecewise linear through the union of their corner times, which reproduces
each event exactly. A wave that combines an arbitrary gradient on the gradient
raster is evaluated at the centres of that raster, across the span of all its
events, and holds its first and last values over the half intervals at its two
ends, which keeps the area of every event whose corners lie on the raster.

A playout holds the waves in waveform memory it sets aside for them on each
gradient axis. How much memory, on what raster, how fast it loads a sample and
how far its loading may run ahead of the playout are its budget
({class}`~pulserver.ir.WaveBudget`). The conversion lays the waves out for the
budget it is given ({func}`~pulserver.ir.plan_waves`), and the cache carries
the layout, with the budget it was made for, where both stages of a playout
read it (`pulseg_get_wave_plan`); a playout with another budget refuses it. A
wave, or a slot, occupies the intervals of the playout's gradient raster that
cover its span, sampled at their centres ({func}`~pulserver.ir.sample_wave`),
on the axes it drives. Where every wave fits at once, each is loaded before the
scan and an instance only selects it. Otherwise each segment position that
plays waves holds a ring of $K$ slots; the $n$-th instance of a segment in the
scan plays slot $n \bmod K$, loaded while the instances before it play. Given
the time the playout takes to load one sample, that loading is checked on the
playout's timeline, scaled by the share of it the loading may take. The scan
starts with its first instance's waves loaded. The waves of each later
instance are loaded one instance after another, from once the instance $K - 1$
before it has started, and have to be loaded before it starts: with two slots,
while the instance before it plays; with more, from further ahead. A chain
whose waves fit neither layout, or an instance of which cannot be loaded in
time, is refused at conversion.

## The two stages of a playout

The C library runs both stages of a playout over a backend the playout
provides (`pulseg_playout_prepare` and `pulseg_playout_scan`, in
`pulseg_playout.h`). The order of the calls and the values they carry are the
library's, so every playout built on it plays a chain alike; what each call
does to the hardware is the backend's.

The first stage needs the definitions and the layout of the waves alone. It
reserves the memory the layout gives the waves, hands the backend each segment
and each of its positions, with the span that position's waves cover and,
where they are streamed, its ring of slots, and loads the waves a resident
layout holds.

The second stage walks the execution stream one segment instance at a time.
It sets the instance's rotation, the prescription's or none under `NOROT`, and
whether it waits for a physiological trigger. Both follow the instance's own
blocks: a virtual segment is shared by instances that differ in either, a
trigger delay and a plain delay for one, so the definition cannot say. For
each block it loads the
block's wave into the slot the instance plays where the waves are streamed,
and sets the block's registers: the RF amplitude, phase and frequency offsets
and shim, the gradient amplitudes or the wave and its amplitudes, the ADC
frequency and phase offsets, and the digital output. Then it starts the
instance.

The receive-gain calibration prescan plays one subsequence from its start to
the segment instance that completes the readouts its sequence declares
(`NumGainCalibrationReadouts`, at least one). It waits for no trigger and
drives no digital output, and every gradient whose amplitude varies across
repetitions, a phase encoding for instance, plays at zero, as does a wave any
of whose logical axes varies.

Waveform memory holds samples of the type and scale a scanner build defines
before it includes the library: `PULSEG_WAVE_SAMPLE`, the type;
`PULSEG_WAVE_FULL_SCALE`, the sample at the peak of a waveform normalised to
unit peak; `PULSEG_WAVE_PHASE_FULL_SCALE`, the phase in radians that sample
plays on a phase waveform; and `PULSEG_WAVE_QUANTIZE`, which turns a scaled
value into a sample. A wave, an arbitrary gradient shape or an RF magnitude is
normalised to unit peak, scaled, and clamped to the full scale
(`pulseg_wave_samples`); its physical scale is the amplitude each block plays
it at. A phase is wrapped into $[-\pi, \pi)$ before it is scaled
(`pulseg_phase_samples`). The defaults are single-precision samples at a full
scale of one, and a phase full scale of $\pi$.

{func}`~pulserver.ir.playout` runs both stages over a backend that plays
nothing and records what each stage hands it, with a model of the waveform
memory. The test suite holds every block the record plays, the events its
position prepared at the amplitudes the scan loop set or the samples its wave
read from memory, to the block {func}`~pulserver.ir.play` resolves; checks
that no load writes over memory the instance in play reads; and compiles both
stages into the 32-bit scanner reader, which plays the same instances and
waves as the host and, built with 16-bit samples, loads the same waves scaled
to that type.

## The heaviest repetition

A scanner evaluates its gradient-heating model on the gradients of a
repetition, and the host's sound pressure check
(`pypulseqpp.safety.check_spl`) chooses the same one. For each subsequence the conversion finds the repetition, of
those the execution stream holds from its first block, over which

$$
E = \int \lVert \mathbf{g}(t) \rVert^2 \, dt
$$

is largest, the earliest on a tie, or the whole subsequence where it does not
repeat, and the cache carries its gradients
({func}`~pulserver.ir.repetition_gradients`, `pulseg_get_tr_corner_points`). $E$ sums, over the blocks and their gradient
events, the square of each event's amplitude times the integral of its
normalised waveform's square, which a rotation leaves unchanged, so the choice
does not depend on the blocks' rotations. The repetition is one the scanner
plays, each block at its own amplitudes and with its own shapes and rotation,
or through its wave: a waveform that plays, not an envelope of
several. Its gradients are joined over its blocks as corner points on one
timeline, along the logical axes and linear in between; the prescription's
rotation takes them to the physical axes, and a model that needs them on a
raster samples them at the centres of its intervals
(`pulseg_sample_corner_points`).

## Cache file

The cache has the name of the first sequence file with its extension replaced:
`.pseg` by default. It is divided into sections that a
consumer loads independently. The pulse-generation stage of a playout reads the
definitions, their waveforms, the waves and their layout, and the gradients of
each subsequence's heaviest repetition; the scan loop also reads
the per-block instances, the rotations and the execution stream, whose size
scales with the scan length. Integer and float fields are 4 bytes. The byte order is recorded
in the file, and a reader on a machine of the other byte order swaps on load. A
cache whose format version differs from the reader's is rejected rather than
read in part.

A cache may be tagged with a vendor code (`PULSEG_VENDOR_*`), which selects the
reader built for that vendor. {func}`~pulserver.ir.summary` reports the
subsequences, segments and readouts of a sequence, computed from the chain or
loaded from a vendor-neutral cache.

## Playback

{func}`~pulserver.ir.play` loads a cache with the C library and walks its
execution stream with the cursor, resolving each block by its own instance:
its duration, the RF and ADC frequency and phase offsets, the RF use, the ADC
window, the gradient amplitudes and the wave, and, on request, the RF and
gradient waveforms each instance plays. With it the cache is compared with the
file it was converted from, without a scanner: the test suite holds every
played block of each fixture to the block its file designs, every wave to the
block's gradients turned by its rotation, and the trajectory the waveforms
trace to the one the file designs ({doc}`virtual-scanner`). It also holds the
waveforms {func}`~pulserver.ir.playout` records through both stages of a
playout to the cursor's, bit for bit.

## Language constraint

The IR is built by C++ passes in `src/cpp/ir/`, which run only on the host:
the segmentation, the waves, their layout in a playout's waveform memory and
each subsequence's heaviest repetition. The cache reader and writer, the
accessors a playout uses to walk the loaded collection, and the two stages of a
playout are compiled into the interpreter, whose vendor toolchains accept ANSI
C. They are therefore C89, in `src/c/`, and call nothing in `src/cpp/`. The test suite
compiles `src/c/` as a scanner build does, 32-bit and vendor-tagged, and reads
back a cache written on the host.

