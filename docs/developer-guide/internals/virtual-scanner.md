# Virtual scanner

How the virtual scanner plays runs of repetitions and to what tolerance, how a
cache is exported to an external Bloch simulator, and what the test suite
establishes by playing caches on it ({doc}`../../explanations/virtual-scanner`).

## Runs of repetitions

Most of a scan repeats a few blocks many times, changing only the phase
offsets of its pulses and ADC events, the amplitudes of its phase-encoding
gradients and the direction of its readout. Such a run plays from each
isochromat's map over one repetition, as
{meth}`~pulserver.virtual.Isochromats.repetitions` plays it, rather than
block by block. A repetition is the fewest consecutive blocks, up to 64, that the
blocks after them repeat event for event, with the same registers but the
phase offsets, the gradients' amplitudes and their waves; runs start at any
block, so a preparation or a train of dummy excitations before the imaging
blocks is a run of its own or plays block by block. Within a run:

- every pulse of a repetition has its phase offset larger by the same
  increment than in the first repetition, and so has every ADC event;
- a gradient that varies across the repetitions differs from the first
  repetition's by a waveform zero during its block's pulse and held through
  its block's ADC window: a phase encoding, whose amplitude varies, or a
  readout turned by a rotation with its prephaser, as a radial spoke is, whose
  wave's corners fall at the first repetition's times; the waveforms of a
  repetition leave no area by any of its pulses, so that they turn only the
  samples of its windows and, by the area they leave over it, each
  isochromat's magnetization at its end;
- or, in the first block, a gradient held through the block, under its pulse,
  varies across the repetitions, as each spoke of a ZTE scan plays its pulse
  under its own readout gradient; the block then reads no ADC window;
- each ADC window is read under a gradient held throughout it.

A run ends where one of these stops holding. Runs of the same blocks join into
one run whose repetitions other blocks play between, as far as their
gradients differ from the first repetition's as the list states: the spokes of
a ZTE scan's shells are one run, with each shell's closing spoke and the ramp
onto the next shell's first spoke between them. The run resumes from the
magnetization those blocks leave, so its maps are made once rather than for
every shell. Stretches of blocks between runs that repeat one another at least
eight times are the repetitions of a run of their own, as those between a ZTE
scan's shells are. An interleaved multislice scan
whose RF spoiling steps each slice's pulse by its own increment, a spin echo
phase-encoded before its refocusing pulse, and spiral readouts, whose
gradients vary during the windows, play block by block; a spiral's windows are
read from the lattice the phantom is sampled on, as
{doc}`bloch-engine` states.

The maps take memory in proportion to the isochromats and to the ADC windows of
a repetition. A repetition whose maps would take more than 4 GiB is not played
from them: a run takes a shorter repetition that fits, or its blocks play one
by one.

{func}`~pulserver.virtual.simulate` and {class}`~pulserver.virtual.Scan` take
the `tolerance` of {meth}`~pulserver.virtual.Isochromats.repetitions`. At zero,
a run samples what its blocks played one by one sample, to the single precision
of the cache's amplitudes, which leave the varying gradients' area by a pulse
at rounding where the run takes it as zero. The console and `pulserver scan`
play runs, and read the windows of the blocks played one by one, to a
tolerance of $10^{-4}$, at which the transients of a steady state are carried
until they fall below it and its fixed points are summed once, by columns of isochromats along the encoded axes; a run whose column sums
would take more than 4 GiB carries every transient instead. Played to a
tolerance, a run also takes as zero the area its varying gradients leave by a
pulse or over a repetition at the six significant digits a Pulseq file keeps
of an amplitude, up to $2 \times 10^{-5}$ of the largest area one of them
plays, and its samples then differ from the blocks' by the phase that area
accrues.

## External simulators

A Bloch simulator that reads Pulseq files, KomaMRI for example, models what
the signal model above leaves out: relaxation, slice profiles and the action
of every RF pulse on the magnetization. A simulation of the design would test
the design alone; {func}`~pulserver.virtual.export` writes the blocks the
cache plays instead, so that a simulation of the file tests the IR as the
virtual interpreter does. Each played block becomes one block of a Pulseq
1.5.1 file, of the duration it plays:

- The RF pulse as the cache holds it: its samples at their times, its
  frequency and phase offsets with their ppm terms resolved at the field the
  cache was converted at, its centre and its use. The channels of a pTx pulse
  are summed, as at unit, in-phase sensitivity. A simulator reads the pulse as
  it reads the design's, and applies the offsets as it applies a design's. In
  Pulseq, a frequency offset and the same phase ramp written into the samples
  play one pulse; KomaMRI simulates the offset in the pulse's rotating frame
  but takes the samples' phase in the opposite sense of rotation, so from the
  ramp it would excite the mirror image of an offset slice. It refers the
  offset's phase to the pulse's centre, which the file therefore records.
- The gradients along the physical axes: those the cache plays, the block's
  rotation in them, turned by $R$ except in blocks labelled `NOROT`, as the
  virtual interpreter turns them. A time-shaped gradient through the corners of all three axes
  carries each, and a step from or to zero at a gradient's first or last
  corner is a ramp 10 ns wide, since a time shape holds one value per time.
  The file therefore needs no rotation extension.
- The ADC window without its offsets. The receiver phase $\theta$ of every
  sample is returned instead, and the simulated samples are demodulated by it
  as the playout demodulates. The offsets act on the samples alone, so no
  simulator has to apply them; KomaMRI applies an ADC's phase offset but
  neither its frequency offset nor a phase modulation.

The file holds the standard sections alone. The text format writes a
gradient's amplitude to six significant figures, and a turned gradient's
amplitude is written as the rotation leaves it, so the k-space of a file
exported under an oblique prescription agrees with the played trajectory to a
relative $10^{-5}$ of its extent.

A job run every night and on demand simulates every fixture and every sequence
pypulseqpp ships with KomaMRI, over one phantom whose density, relaxation times
and off-resonance vary across it: as designed, and as exported from its cache.
KomaMRI drops the ppm term of an offset, so a design file carrying one is given
to it as Pulseq 1.4.1, whose writer resolves the term; and its rotation of a
block can drop a corner a gradient holds twice, such as the peak of a trapezoid
without a flat top, so the job turns a design's gradients itself. The two
simulations then differ only where the cache plays something other than the
design, or by the rounding of the text format.

The same job compares KomaMRI with the virtual scanner's Bloch simulation.
KomaMRI adds a pulse's phase shape and phase offset to its field with the
opposite sign, and refers the phase its frequency offset accrues to the
pulse's centre ({doc}`bloch-engine`). It also joins a pulse's samples linearly and takes the field at the
start of each time step, where the playout holds each sample of a pulse sampled
at the middles of equal intervals over its interval; read as written, such a
pulse would play half an interval late. The exported file is therefore
simulated a third time with each pulse rewritten in KomaMRI's convention: its
samples conjugated, each sample of a pulse sampled at the middles of equal
intervals held from the start of its interval, and its phase offset $\phi$
replaced by $-\phi - 2\pi f t_c$, for the frequency offset $f$ and the centre
$t_c$. KomaMRI then plays the field the virtual scanner plays. That signal,
demodulated by the receiver phase the export returns, is compared with the
virtual scanner's simulation of the cache on the same spins.

## What a run establishes

- The trajectory the cache plays is the one each file designs, for the
  fixtures and for every sequence pypulseqpp ships, at small sizes, and under
  an oblique and a reflected prescription the one `pypulseqpp.TransformFOV`
  turns the design into for the checks. A block labelled `NOROT` plays turned
  by its own rotation alone.
- The cache carries the RF centre a design records where it is away from the
  magnitude peak, and plays every RF pulse of a fat-saturated EPI and of a
  spin echo as the design draws it.
- Off resonance, every sample of the fixtures and of every shipped sequence
  accrues the phase its file's excitation, refocusing and ADC timing give it,
  so the cache plays each echo as the design times it.
- Fat precesses at its chemical shift at the magnet's field. A fat saturation
  leaves water and fat what pypulseqpp's relaxation-free Bloch simulation of
  the designed pulse leaves them, and one converted at another field misses
  the fat.
- The enrichment states the trajectory of the identity, along the logical
  axes, for every readout.
- An object posed where an axial, an oblique or a reflected prescription
  places the field of view is acquired by every shipped sequence as it is at
  the isocentre, EPI and spiral readouts, whose curvature the proxy's phase
  completes, and readouts turned by a rotation extension included; an object
  away from the offset, or turned otherwise than prescribed, is not.
- A series streamed through the proxy under an axial, an oblique and a
  reflected prescription is reconstructed into the image of the phantom at the
  isocentre, and the image carries the prescribed centre and the columns of
  $R$ as its read, phase and slice directions; a series short of a readout is
  refused.
- Played on isochromats, the cache of every fixture samples what its design's
  blocks played one by one sample, under an axial, an oblique and a reflected
  prescription, with the magnetization carried across the files of a chain.
- The runs of a balanced SSFP, a spoiled and a multi-echo gradient echo, a
  spin echo, a fast spin echo and an MPRAGE sample what their blocks played
  one by one sample; a balanced SSFP plays as runs of one repetition but for
  its preparation, and to a tolerance of $10^{-4}$ samples within a thousandth
  of its exact samples, with its fixed points summed by columns and without.
  An interleaved multislice scan whose pulses step unevenly, a spin echo
  phase-encoded before its refocusing pulse, and spiral readouts play block by
  block, and so do the echoes of a three-echo gradient echo where the maps of
  one window fit and those of three do not. A balanced gradient echo whose
  rewinders are off by a few millionths plays as one run to a tolerance. A run
  played across spans samples what it samples played whole.
- The spokes of a ZTE scan of ten shells play as one run, each spoke's pulse
  read off its tables, resumed after the blocks between its shells, which play
  as a run of their own; the scan samples what its blocks played one by one
  sample, played whole and in spans that end within a shell.
- The phantom sampled as isochromats, posed where an axial, an oblique or a
  reflected prescription places the field of view and scanned by a
  single-shot EPI, is acquired as the analytic phantom is: a 90° excitation
  leaves its density at the phase the signal model states. Each ellipse's
  isochromats relax with its $T_1$ and $T_2$ and precess at its chemical shift
  at the magnet's field.
- Played in spans, the scan of every fixture plays each block once: the
  spans' readouts are those of the whole scan, and the sound of a single-file
  fixture, joined across its spans, is `Sequence.sound` of its design as the
  checks turn it, under an axial, an oblique and a reflected prescription. A
  span played at a speed is released once the clock has passed it, and a
  series streamed readout by readout is reconstructed as the same series sent
  whole. A scan simulated twice as slowly as it plays starts its clock late
  enough that no span holds it; a span simulated after its time holds the
  clock, and the spans after it keep its pace.
- `pulserver scan` records the series the virtual scanner acquires of an
  imported file, sample for sample, and streams a generated design to a
  reconstruction proxy, whose image carries the prescribed centre and
  directions.
- The exported file of every fixture and every shipped sequence, read and
  integrated by pypulseqpp, has the trajectory the cache plays, under an
  axial, an oblique and a reflected prescription, and holds each RF pulse with
  the samples, offsets, centre and use of the design's.
- In the scheduled KomaMRI job, the signal simulated from the exported file of
  every fixture and every shipped sequence is the one simulated from its
  design, to the rounding of the text format, and the virtual scanner's Bloch
  simulation of the cache gives the signal KomaMRI simulates of the exported
  file in its own RF convention, to the difference of the two simulators' time
  steps through an RF pulse.

