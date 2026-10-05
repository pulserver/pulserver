# IR cache

The implementation of the scanner representation
({doc}`../../explanations/scanner-representation`): how gradients that a
prepared segment cannot play are carried as waves, how a playout loads the
cache in two stages, the gradients kept for the scanner's heating and acoustic
models, and the layout of the cache file. The C headers an interpreter
includes are in
[`src/c/include/pulseg/`](https://github.com/pulserver/pulserver/tree/main/src/c/include/pulseg).

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
blocks: a segment definition is shared by instances that differ in either, a
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

A scanner evaluates its gradient-heating and acoustic models on the gradients
of a repetition. For each subsequence the conversion finds the repetition, of
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

