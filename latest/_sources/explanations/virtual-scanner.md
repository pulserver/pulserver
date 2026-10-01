# The virtual scanner

The IR, the raw-data header contract and the reconstruction path are tested
against data a scanner acquires. Data sampled at the k-space locations the
enrichment computes cannot test the enrichment, and a design compared only
with itself cannot test the IR. The virtual scanner stands in for the scanner
and nothing else: the design calls, the IR, the reconstruction proxy and the
reconstruction plugins it drives are the production code. Its data are
acquired along the trajectory the cache plays, from an analytic phantom or by a
Bloch simulation of the blocks the cache plays, and sent as the scanner's
reconstruction client sends them.

## Stand-ins

| Stand-in | Replaces | Built from | Exercises |
| --- | --- | --- | --- |
| Design call | The interpreter host process's call | `pulserver design generate`, or {func}`~pulserver.host.call` | Request and reply blocks, presets, design errors, check failures |
| Virtual interpreter | The playout | The C library's two playout stages over a recording backend, {func}`~pulserver.ir.playout` | Segmentation, the events each segment position is prepared with, the registers the scan loop sets on each block, the rotation and trigger of each segment instance, the waves and the waveform memory they are loaded into |
| Physics | Magnet, coils and subject | {class}`~pulserver.virtual.Phantom`, {func}`~pulserver.virtual.acquire`, {func}`~pulserver.virtual.simulate` | The trajectory the enrichment has to state, the demodulation of the prescription, the timing of every echo, the RF frequencies ppm offsets resolve to; in the Bloch simulation, every RF pulse, gradient and receiver phase the cache plays |
| Scan clock | The scanner's acquisition in real time, and its gradient coils' sound | {class}`~pulserver.virtual.Scan` | The rate at which readouts reach the reconstruction; the sound of the gradients the cache plays |
| Reconstruction client | The scanner's reconstruction client | {func}`~pulserver.virtual.send` | The header and acquisition contract of {doc}`../user-guide/reconstruction-client` |

## Played trajectory

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

{func}`~pulserver.virtual.trajectory` integrates those gradients into the
k-space location $\mathbf{k}$ of every ADC sample, in 1/m along the physical
axes. The rotation $R$ of the prescription, from logical to physical axes, is
not in the cache ({doc}`ir-cache`): the virtual interpreter is given it, as a
playout is, and turns each block's gradients, which carry the block's own
rotation, by $R$, except in the segment instances whose blocks are labelled
`NOROT`, which play as the cache holds them. This is the frame
{func}`~pulserver.ir.check` checks in. Under
the identity the physical axes are the logical ones. An excitation returns
$\mathbf{k}$ to zero, and a refocusing pulse negates it, at the RF centre the
design records, which the cache carries for each RF event with its use; each
file of a chain starts from zero. The test suite holds this trajectory to the
one each file designs, turned as `pypulseqpp.TransformFOV` turns it for the
checks, and the trajectory of the identity to the one the enrichment states.

## Signal model

The phantom lies in the physical frame, where a rotation and a position place
it ({class}`~pulserver.virtual.Phantom`). Each of its ellipses holds spins of
one chemical shift $\sigma$, in ppm from water, which precess at
$f_\sigma = 10^{-6}\sigma\gamma B_0 + \Delta f$ from the scanner's centre
frequency: the shift resolved at the magnet's field $B_0$ with pypulseqpp's
default $\gamma$, in Hz/T, plus an off-resonance $\Delta f$ common to every
spin.

After an excitation of RF phase $\phi_e$ at its centre, the magnetization is
transverse, at the phantom's density $\rho_\sigma(\mathbf{r})$ whatever the
flip angle, with phase $\pi/2 - \phi_e$, and it precesses as
$\exp(-2\pi i\,(\mathbf{k}\cdot\mathbf{r} + f_\sigma\tau))$, with
$\mathbf{r}$ and $\mathbf{k}$ along the physical axes and $\tau$ the time it
has precessed freely: counted from the excitation's centre and, as
$\mathbf{k}$ is, negated about the centre of each refocusing pulse, so that a
spin echo refocuses it. A pulse of phase zero turns the magnetization from
$+z$ towards $+y$, as it turns the magnetic moment of a nucleus of positive
gyromagnetic ratio. A refocusing pulse of phase $\phi_r$ conjugates the
magnetization about $-\phi_r$. The RF phase at the centre is the pulse's
phase offset plus its frequency offset times the time since the pulse began.
Coil $c$, of sensitivity $s_c(\mathbf{r})$, receives

$$
S_c(t) = e^{i\psi} \sum_\sigma m_\sigma \int \rho_\sigma(\mathbf{r})\,
s_c(\mathbf{r})\, e^{-2\pi i\,(\mathbf{k}(t)\cdot\mathbf{r} + f_\sigma\tau(t))}\,
d\mathbf{r},
$$

with $\psi$ the phase the RF pulses left and $m_\sigma$ the longitudinal
magnetization of shift $\sigma$ the excitation tipped, as a fraction of
equilibrium. A saturation pulse scales $m_\sigma$ by the $z$ component it
leaves at $f_\sigma$, from $+z$, in pypulseqpp's relaxation-free Bloch
simulation (`pypulseqpp.sim_bloch`) of the pulse the cache plays, in the frame
of its frequency. The next excitation tips what remains, and the
magnetization is at equilibrium again after it. A saturation pulse played
under a gradient saturates a band in space, which a species of the phantom
does not hold, and is refused. The playout demodulates $S_c$ by
$\exp(i\theta(t))$, where the receiver phase $\theta$ is the ADC phase
offset at the ADC's start, advancing at its frequency offset, plus its phase
modulation.

These are the conventions under which the field-of-view translation applied
when the IR is built ({doc}`ir-cache`) recentres an object. Where every block
is turned by $R$, $\mathbf{k}$ is $R\mathbf{k}_L$, with $\mathbf{k}_L$ its
value along the logical axes. An object displaced to the prescribed centre
$R\mathbf{d}$, for the offset $\mathbf{d}$ along the logical axes, gains the
phase $-2\pi\,(R\mathbf{d})\cdot(R\mathbf{k}_L) = -2\pi\,\mathbf{d}\cdot\mathbf{k}_L$,
which the demodulation removes, and an object turned by $R$ presents at
$R\mathbf{k}_L$ its own transform at $\mathbf{k}_L$. An object posed at the
prescribed centre, its axes turned by $R$, is therefore acquired as the same
object at the isocentre under the identity, a reflection in $R$ included. The
phantom's ellipses and its sensitivities, sums of plane waves, have analytic
transforms, so $S_c$ is evaluated exactly at every sample.

Relaxation, diffusion, slice profiles and every RF use other than
excitation, refocusing and saturation are not modelled. Without relaxation, a
saturation pulse leaves spins at its own frequency at the cosine of its flip
angle: pypulseqpp's `FatSaturation`, of 110° by default, leaves fat at
$\cos 110° \approx -0.34$ of its magnetization. A readout before the first
excitation of its file acquires zeros.

## Bloch simulation

The signal model leaves out what an RF pulse does beyond an ideal excitation,
refocusing or saturation, and what the magnetization does between them.
{func}`~pulserver.virtual.simulate` plays the blocks
{func}`~pulserver.ir.playout` returns on isochromats instead
({class}`~pulserver.virtual.Isochromats`, {doc}`bloch-simulation`): flip
angles, slice profiles, relaxation and the coherences one repetition leaves to
the next act as the Bloch equation gives them, and the magnetization is carried
from each block to the next, across the files of a chain. Each block's
gradients are those of the played trajectory, turned by $R$ except in blocks
labelled `NOROT`. Its RF pulse and its ADC play as the RF and ADC events of a
Pulseq block, with the frequency and phase offsets the playout sets, and each
sample is demodulated by $\exp(i\theta(t))$ as above. A 90°
excitation of phase $\phi_e$ leaves the magnetization at the phase
$\pi/2 - \phi_e$ of the signal model.

{meth}`~pulserver.virtual.Phantom.isochromats` samples the phantom on a square
grid aligned with its axes. Each ellipse contributes the points of the grid
inside it, each an isochromat of proton density the ellipse's intensity times
the area of a grid cell, relaxing with the ellipse's $T_1$ and $T_2$ and
precessing at its $f_\sigma$; the coil sensitivities are sampled at the same
points. The sum over the isochromats approximates each ellipse's transform
below the grid's Nyquist frequency, $1/(2\Delta)$ for a spacing $\Delta$, and
converges to it as the grid is refined, so a grid several times finer than the
image's pixel acquires the analytic phantom wherever the signal model holds.

An isochromat that no excitation pulse tips from rest adds nothing to the
readouts, so a scan is simulated on the isochromats its excitation pulses
excite. {func}`~pulserver.virtual.excited` finds their slabs: each pulse the
cache labels an excitation is played, at its nominal amplitude and at twice
it, on a line of isochromats along its gradient, turned as the scan turns it,
and its slab holds the fields at which it changes the magnetization at rest by
at least a hundredth of the most it changes it, sidelobes included. An
isochromat at $\mathbf{r}$ precessing at $f$ sees the field
$\mathbf{g} \cdot \mathbf{r} + f$ during a pulse played under the gradient
$\mathbf{g}$, so a chemical shift or the subject's field moves an isochromat's
slab as it moves the slice, and each slice of a multislice scan is a slab of
its own. A pulse played without a gradient, or under one that changes during
it, excites the whole phantom, as a volumetric excitation does. So do pulses
played under more than 16 gradients, as a ZTE scan plays one under each
spoke's: between them, their slabs leave out next to none of the object. What other
pulses tip into the transverse plane outside the slabs, such as the free
induction decay of an imperfect refocusing pulse, is left out. The phantoms'
`isochromats` keep those a region, such as {class}`~pulserver.virtual.Slabs`,
answers for.

Played on isochromats, the cache samples what the design samples when its
blocks, turned as the checks turn it, are played one by one on the same
isochromats with the gradients `pypulseqpp.Sequence.waveforms` reports, to the
single precision of the cache's waveforms. A gradient turned by a block's
rotation is stored as the rotation leaves it, in single precision, and in a
sequence that leaves its transverse magnetization unspoiled from one repetition
to the next, the phase that rounding accrues is what separates the two.

### Runs of repetitions

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
{doc}`/explanations/bloch-simulation` states.

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

## Coils and the subject's field

The coils of the signal model are the phantom's own. A scan on the Bloch
simulation can take one of the scanner's {obj}`~pulserver.virtual.COILS`
instead, fixed in the physical frame, each a transmit coil and a receive coil,
named `transmit/receive`: the body coil both ways, taken to be one channel of
unit sensitivity each way; the body coil transmitting to a 48-channel receive
head array, `body/head48`; and an 8-channel head coil for parallel transmission
with a 32-channel receive head array, `head8/head32`. The head coils'
sensitivities are BART's coil models rather than electromagnetic simulations:
the 8-channel coil is `HEAD_2D_8CH`, constant along $z$, and the arrays are the
first 48 and 32 channels of `HEAD_3D_64CH`, each sampled by bartorch over a cube
25.6 cm wide about the isocentre and interpolated linearly between the samples.
The receive sensitivities $s_c$ have a root sum of squares of 1 at the
isocentre. The transmit sensitivities $s^+_c$ are the complex conjugates of the
8-channel coil's receive sensitivities, the quasi-static limit of reciprocity;
the models show none of the dielectric effects of a wavelength comparable to
the head.

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

Each transmit channel plays the pulse the cache holds for it, scaled at each
isochromat by $s^+_c(\mathbf{r})$, and the channels' fields add. A pTx pulse
holds one waveform per channel. A single-channel pulse plays on every channel,
weighted by its block's RF shim or, without one, by the coil's default shim,
the unit weights that bring the channels into phase at the isocentre, where the
pulse then has its nominal amplitude. A shim of another number of channels is
refused.

The maps' transmit coils come with the VOPs mariepy compresses from the same
fields, and every design is made under those of the exam's transmit coil: its
limits name the VOP file, the default shim, and the drive of every channel per
hertz of a pulse's amplitude with which the channels' fields reach that
amplitude at the isocentre, $2 / (\gamma \sum_c |B^-_c(\mathbf{0})|)$ in the
maps' unit of drive. The IR cache then reports each subsequence's SAR against
the reference pulse in that coil ({doc}`ir-cache`).

{class}`~pulserver.virtual.BrainWeb` carries the field its own susceptibility
adds to $B_0$. Its head is water, of volume susceptibility $-9.05$ ppm, in air
of $0.36$ ppm (Schenck, Med Phys 23:815, 1996), and the field along $B_0$ is
the susceptibility difference convolved with the dipole kernel
$1/3 - k_z^2/|\mathbf{k}|^2$, whose $1/3$ is the Lorentz sphere's (Marques and
Bowtell, Concepts Magn Reson B 25:65, 2005). The constant and linear terms over
the head are removed, as a first-order shim removes them. The field is in ppm
of $B_0$, so an isochromat precesses $\gamma B_0$ times it faster, whatever
the magnet's field.

## Scan clock and sound

A scanner acquires in real time: each readout reaches the reconstruction once
the scanner has played it, and the gradients sound as they play.
{class}`~pulserver.virtual.Scan` plays the cache on isochromats against a scan
clock, the sum of the durations of the blocks played, in spans of whole blocks
that end where a repetition of a run starts. Each span carries the readouts of
its blocks, as
{func}`~pulserver.virtual.simulate` returns them, and the sound of the gradients
it plays. At a speed, a span is released once the wall clock, running that many
times as fast as the scan, has passed its end, so that a reconstruction
receives the readouts at the rate a scanner acquires them;
{func}`~pulserver.virtual.send` sends each readout as it is released.

The spans are simulated in a thread of their own, ahead of their release, and
the clock starts once the simulation will stay ahead of it to the end of the
scan. A span's simulation time is estimated from the spans simulated before it:
per ADC sample for a span that acquires, since the readouts' coil sums dominate
it, and per second of scan time for one that does not, such as a train of
dummy excitations. Where the simulation runs faster than the scan, the clock
starts once a span of each kind has been simulated; where it runs slower, as
for a short-TR balanced SSFP on a head of many isochromats and coils, most of
the scan is simulated before the clock starts and the rest while it runs. A
span simulated after its end on the clock, where the estimate fell short,
holds the clock until it is, and the spans after it keep the scanner's pace.

The sound is MATLAB Pulseq's, from `pypulseqpp.gradient_sound`: the gradients
along the physical axes, the x axis on the left channel, the y axis on the
right and half of the z axis on both, smoothed by MATLAB's Gaussian window of
$2\,\mathrm{round}(f_s/6000) + 1$ samples at the sample rate $f_s$, and scaled
so that the loudest sample of the scan is 0.95. The window reaches past the
ends of each span into the gradients on either side, so the spans' sounds,
joined, are the sound of the whole scan: that of `Sequence.sound` of the design
under the same prescription, to the single precision of the cache.

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
pulse's centre ({doc}`bloch-simulation`). It also joins a pulse's samples linearly and takes the field at the
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
  the isocentre, EPI and spiral readouts, which the translation gives a phase
  modulation, and readouts turned by a rotation extension included; an object
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

## See also

* {doc}`ir-cache` — the IR, its prescription and its playback.
* {doc}`../user-guide/reconstruction-client` — the stream the virtual reconstruction client sends.
* {doc}`../api/virtual` — the phantom, the acquisition, the Bloch simulation, the scan clock, the client and the export.
* {doc}`bloch-simulation` — the Bloch simulation the cache is played with, and how it plays each Pulseq event.
