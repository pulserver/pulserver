# Virtual scanner

The implementation of the virtual scanner
({doc}`../../explanations/virtual-scanner`): how the virtual interpreter
resolves each block's waveforms, how the Fourier engine builds its timeline,
event streams, bases and images and to what tolerance, how a cache is exported
to an external Bloch simulator, what the test suite establishes by playing
caches, the coils taken from field maps, and the scan clock and the sound.

## Played waveforms

{func}`~pulserver.ir.playout` plays the cache through the two stages of a
playout: the first prepares the events of each segment position, and the
scan loop sets the registers of each block of each segment instance
({doc}`ir-cache`). With `waveforms`, it returns what each block plays, timed
from the block's start. Its gradients are the events its position is prepared
with, through the position's representative instance, the one of largest
energy (`pulseg_get_grad_amplitude` and `pulseg_get_grad_time_us`), at the
amplitudes the scan loop sets. At a position that plays waves, one whose
blocks carry a rotation or play a shape the prepared events do not hold, as
the interleaves of a spiral drawn as distinct shapes do, they are the wave
the scan loop selects, which `pulseg_materialize_wave` returns, at the wave's
amplitudes. Its RF pulse is the magnitude and phase shapes its position is
prepared with, which `pulseg_get_rf_magnitude` and `pulseg_get_rf_phase`
return, the phase in cycles as Pulseq stores it, at the block's amplitude.
A wave plays as the IR defines it, linear between its points; how a
playout's hardware plays the samples it loads on its own raster is not
modelled. {func}`~pulserver.ir.play` walks the same cache with the cursor and
resolves each block by its own instance, and the test suite holds the two to
the same waveforms, bit for bit.

## Fourier engine

{class}`~pulserver.virtual.FourierPlayer` builds the acquisition of a tissue
once, from the whole playout, and then acquires any range of readouts from it.
`src/pulserver/virtual/_fourier.py` holds the engine and
`src/pulserver/virtual/_timeline.py` the timeline it reads; the integration of
the gradients and the search for each readout's echo are native, in
`src/cpp/fourier/`.

### Timeline

The timeline holds the moment of the played gradients from the start of the
scan along the logical axes, without the resets of an excitation: a block's
gradients carry its own rotation, and those of a block labelled `NOROT`, which
play along the physical axes, are turned back by $R$. An excitation sets the
origin $k$ is measured from and a refocusing pulse negates $k$, at the RF
centre the cache carries, and each file of a chain starts from no
excitation.

Each RF pulse is reduced to a flip angle and the phase of its axis, from
`pypulseqpp.sim_bloch` of its samples on a uniform raster with the channels of
a pTx pulse summed, and to a profile: the angle it tips magnetization at rest
through, over the angle on resonance, at detunings $1/(16T)$ apart within
$\pm 64/T$ of its frequency, for a pulse of duration $T$, cut where it falls
below $1/128$. Pulses of the same samples at the same amplitude share one
profile. A pulse plays under the gradient read at five points of its duration
when the gradient changes by less than $10^{-6}$ of its largest axis. A
gradient that changes but keeps one direction across the pulse's raster, as
the alternating lobes of a spectral-spatial pulse do, gives the pulse a profile
across both the field and the position along that direction: a Bloch
simulation of its samples under that gradient, over the box of fields and
positions the tissue's cubes span, at 8 points per resolution ($1/T$ in
frequency, the inverse of the span of its excitation k-space in position), at
most 256 along either, read bilinearly. A pulse under a gradient that turns
selects by frequency alone.

Each readout's echo is the sample at which the pathway it reads passes nearest
the centre of k-space, found first among 64 samples spread over the window
and then among those around the nearest. The pathways considered are the free
induction and the echoes of the one or two intervals before; a readout whose
excitation's interval winds less than `SHIFT_CYCLES`, half a cycle,
across a voxel reads the free induction.

### Groups, stations and shifts

The voxel along each logical axis is the smallest of $1/(2k_\mathrm{max})$,
the full width at half maximum of each excitation's profile over its gradient
along that axis (on resonance across the position, for a profile in both),
and the extent of the tissue. A pulse's selector is its
gradient, its frequency and its profile, and the direction and samples of a
profile in both field and position; where the pulses play under more than
16 distinct gradients, as a ZTE scan's do, every selector selects by frequency
alone. Under each selector, a cube's flip angle relative to the one on
resonance is rounded to a level: halving from $2^{-7}$ to $2^{-4}$, then in
steps of 0.05 from 0.1 to 2. Cubes of the same levels under every selector form
a group, and the groups that together hold less than $10^{-3}$ of the excited
density join the kept group nearest in level that the same selectors turn.
Cubes no excitation turns are dropped.

An interval between two events shifts the states by one order where the
median of its dephasing across the voxel, over the intervals at the same pair
of segment positions, reaches `SHIFT_CYCLES`. The part of an
interval's moment that changes from one repetition to the next counts where
the intervals between two consecutive pulses do not sum it to less than that.

### Event streams

A station's stream holds the pulses whose selectors turn its groups and the
readouts its excitations precede, each readout at its echo. Each event is
described to blochsim as an ideal rotation through its selector's flip angle
at each group's level, and the stream is simulated by `EpgEngine` for each
atom: a class of tissue, a bin of the field and a bin of the transmit field's
magnitude. A cube's signal interpolates linearly between the two bins of each
that bracket its frequency and its transmit magnitude, four atoms in all. The
transmit field is binned in steps of 0.05 of the nominal amplitude. The field
is binned only where an interval between two pulses shifts nothing; the bin is
the narrower of the width over which the phase changes by 0.3 rad over the
longest time from an excitation to an echo and a 32nd of the inverse of the
shortest such interval, and lies between 0.5 Hz and 16 Hz.

A stream whose pulses recur, with the same phase increments, times and
shifts, every $q$ of them, plays its first periods until the deviation from
the periodic state, which relaxation contracts by at least
$\exp(-t/T_{1,\mathrm{max}})$, falls below $10^{-3}$ of the equilibrium
magnetization, and its readouts in later periods record what the same readout
of the last period played records. Up to 64 periods are tried, from those at
which the middle pulse of the stream recurs.

### Bases and images

The signals of a station's groups and atoms at its readouts' echoes are
spanned by the fewest leading singular vectors that leave a relative residual
of `TOLERANCE`, at most
`MAX_TERMS` of them; where the groups and
atoms are too many to simulate at once, the basis is fitted to a random sample
of them and the rest are projected onto it. The decay, $T'_2$ dephasing and
precession across the readouts are tabulated per pair of $T_2$ and $T'_2$ at
frequencies at most 0.5 Hz apart and spanned, by a randomized range finder, to
the same residual; a cube's coefficients interpolate linearly in frequency.

The pairs of temporal and readout terms are compressed to the fewest
combinations that span the weights of up to $2^{16}$ cubes of the station to
the same residual, and each combination is an image. The image grid spans the
logical axes the trajectory encodes across the cubes, at least two, with
frequencies reaching 1.25 times past the widest $k$ and at least 16 points
along each axis. Where the cubes lie on a lattice finer than that, the grid is
the lattice itself: each image holds the cubes' weights at their centres, and
each sample is weighed by the cube's spectrum, a product of sincs along its
edges, with its $k$ folded into the period the lattice's spectrum repeats over.
Elsewhere, the cubes are spread onto the grid by the adjoint of bartorch's
NUFFT and multiplied by the cube's spectrum at the grid's frequencies. The
receive sensitivities are evaluated at points at most 4 mm apart over the grid
and interpolated linearly onto it. Each image times each coil's sensitivity is
transformed to the samples by bartorch's NUFFT, its centre's phase and the
receiver phase relative to the echo's applied to each sample.

Readouts are acquired in batches that run ahead of the range asked for,
growing from $2^{18}$ samples to $2^{21}$, since a batch costs one transform of
the images whatever its samples.

## External simulators

A Bloch simulator that reads Pulseq files, KomaMRI for example, integrates the
Bloch equation through every RF pulse, which the Fourier engine reduces to a
rotation at its centre. A simulation of the design would test
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
- The Fourier engine's timeline samples the trajectory the cache plays, for
  the fixtures under an axial, an oblique and a reflected prescription. Each
  readout's echo is its sample nearest the centre of k-space, and each readout
  reads the pathway that passes the centre during it.
- A spoiled train shifts the states after each readout and nowhere else, and a
  flat phantom is not dephased along the axis it does not span. Each slice of
  an interleaved multislice scan is a station of its own, and a pulse played
  without a gradient selects by frequency: a fat saturation turns the fat and
  not the water.
- Of a phantom filling its slices, of an unspoiled and of a balanced steady
  state, the Fourier engine acquires the signals of a Bloch simulation of
  isochromats held as reference fixtures, to a few per cent of their norm.
- A stream plays its repetitions until they settle and reads the rest off the
  last; a cube lattice finer than the image grid is read through the cubes'
  spectrum; and a group of flip angles holding too little of the density joins
  the nearest one the same pulses turn.
- Played in spans, a gradient echo and an EPI play each block once: the
  spans' readouts are those of the whole scan, and their sound, joined across
  spans and stretches, is `Sequence.sound` of the design as the checks turn it,
  under an axial, an oblique and a reflected prescription. A scan simulated in
  one stretch is released in spans of the length asked for. A span played at a
  speed is released once the clock has passed it, and a series streamed
  readout by readout is reconstructed as the same series sent whole. A scan
  simulated twice as slowly as it plays starts its clock late enough that no
  span holds it; a stretch simulated after its time holds the clock, and the
  spans after it keep its pace.
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
  design, to the rounding of the text format.

## Coils from field maps

A console given field maps takes the same coils from electromagnetic
simulations of BrainWeb's head instead. mariepy, a port of MARIE 3.0, solves
the head in a quadrature birdcage, whose two linear modes are the body coil's
two channels, in the 8-channel coil and in the two arrays, and writes each
channel's circular components $B^\pm_c = \mu_0 (H_x \pm j H_y)$ over the head,
for the time dependence $e^{+j\omega t}$. With $B_0$ along $+z$, the part of a
channel's field that rotates with the magnetization is half the complex
conjugate of its $B^-_c$, and what it receives is weighted by the complex
conjugate of its $B^+_c$: $s^+_c \propto \overline{B^-_c}$ and
$s_c \propto \overline{B^+_c}$, scaled as the models' are. Outside the head
each map takes the value of the nearest voxel inside it, and between voxels it
is interpolated linearly. The maps show the dielectric effects the models do
not; they are solved in BrainWeb's head alone, so such a console examines
BrainWeb whatever the subject is called.

The maps' transmit coils come with the VOPs mariepy compresses from the same
fields, and every design is made under those of the exam's transmit coil: its
limits name the VOP file, the default shim, the drive of every channel per
hertz of a pulse's amplitude with which the channels' fields reach that
amplitude at the isocentre, $2 / (\gamma \sum_c |B^-_c(\mathbf{0})|)$ in the
maps' unit of drive, and the head and local SAR limits of IEC 60601-2-33's
normal operating mode. The IR cache then reports each subsequence's SAR against
the reference pulse in that coil ({doc}`../../explanations/designs`).

{class}`~pulserver.virtual.BrainWeb` carries the field its own susceptibility
adds to $B_0$. Its head is water, of volume susceptibility $-9.05$ ppm, in air
of $0.36$ ppm (Schenck, Med Phys 23:815, 1996), and the field along $B_0$ is
the susceptibility difference convolved with the dipole kernel
$1/3 - k_z^2/|\mathbf{k}|^2$, whose $1/3$ is the Lorentz sphere's (Marques and
Bowtell, Concepts Magn Reson B 25:65, 2005). The constant and linear terms over
the head are removed, as a first-order shim removes them. The field is in ppm
of $B_0$, so a cube precesses $\gamma B_0$ times it faster, whatever
the magnet's field.

## Scan clock and sound

A scanner acquires in real time: each readout reaches the reconstruction once
the scanner has played it, and the gradients sound as they play.
{class}`~pulserver.virtual.Scan` acquires the cache by the Fourier engine
against a scan clock, the sum of the durations of the blocks played, and
releases it in spans of whole blocks that last at least the length asked for.
The engine is built when the scan is, before its first span. Each span carries
the readouts of its blocks, as {func}`~pulserver.virtual.simulate` returns
them, and the sound of the gradients it plays. At a speed, a span is released
once the wall clock, running that many times as fast as the scan, has passed
its end, so that a reconstruction receives the readouts at the rate a scanner
acquires them; {func}`~pulserver.virtual.send` sends each readout as it is
released.

The scan is simulated in a thread of its own, ahead of its release, a stretch
at a time: a stretch lasts at least the length asked for and ends at the first
block after which the samples acquired since the start of the scan pass a
multiple of $2^{18}$, or at the end of the scan, so that each costs about one
transform of the images. A stretch is released in spans, the last of which
takes the stretch's tail shorter than the length asked for. The clock starts
once the simulation will stay ahead of it to the end of the scan: each stretch
is due on the clock at its start. A stretch's simulation time is estimated
from the stretches simulated before it: per ADC sample for a stretch that
acquires, since the transforms of its samples dominate it, and per second of
scan time for one that does not, such as a train of dummy excitations, at the
rate of the stretches that acquire until one that does not has been
simulated. Where the simulation runs faster than the scan, the clock starts
once the first stretch that acquires has been simulated; where it runs slower,
as for a head received by many coils on a CPU, most of the scan is simulated
before the clock starts and the rest while it runs. A stretch simulated after
its start on the clock, where the estimate fell short, holds the clock there
until it is, and the spans after it keep the scanner's pace.

The sound is MATLAB Pulseq's, from `pypulseqpp.gradient_sound`: the gradients
along the physical axes, the x axis on the left channel, the y axis on the
right and half of the z axis on both, smoothed by MATLAB's Gaussian window of
$2\,\mathrm{round}(f_s/6000) + 1$ samples at the sample rate $f_s$, and scaled
so that the loudest sample of the scan is 0.95. The window reaches past the
ends of each span into the gradients on either side, so the spans' sounds,
joined, are the sound of the whole scan: that of `Sequence.sound` of the design
under the same prescription, to the single precision of the cache.
